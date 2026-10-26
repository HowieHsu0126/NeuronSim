// Distance-dependent repeated Bernoulli projections on a periodic square.
// cuRAND Philox streams are independent of target batching; weights use a
// separate subsequence from topology. This reproduces a probability law, not
// NEST's RNG sequence. Compile with --fmad=false and without --use_fast_math.
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace {
void check(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
template<class T> struct Buffer {
    T* data = nullptr;
    Buffer() = default;
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    ~Buffer() { if (data) cudaFree(data); }
    void allocate(size_t n) {
        if (data) { check(cudaFree(data)); data = nullptr; }
        if (n) check(cudaMalloc(reinterpret_cast<void**>(&data), n * sizeof(T)));
    }
    void copy(const T* host, size_t n) {
        allocate(n);
        if (n) check(cudaMemcpy(data, host, n * sizeof(T), cudaMemcpyHostToDevice));
    }
};
struct Edge {
    int64_t source, target;
    double weight, raw_delay;
    int32_t repeat, anchor_index;
};
static_assert(sizeof(Edge) == 40, "Python edge ABI must match");
struct Parameters {
    double extent, radius, beta, p0, weight_mean, weight_sd, delay_offset, speed, origin_x, origin_y;
    uint64_t seed, stream;
    int cells, repeats, source_count, target_count;
};
struct Context {
    Parameters p;
    Buffer<double> sx, tx;
    Buffer<int64_t> sid, tid;
    Buffer<int32_t> cell_offsets;
};
__device__ double modulo(double value, double extent) {
    double result=fmod(value,extent);
    return result<0 ? result+extent : result;
}
// NEST masked_iterator wraps the mask's lower corner, then visits additional
// anchors where its upper corner crosses the tree extent. It does not dedup IDs.
__device__ double primary_anchor(double target, double origin, const Parameters& p) {
    return modulo(target-p.radius-origin,p.extent)+p.radius+origin;
}
__device__ int grid_cell(double position, double origin, const Parameters& p) {
    return min(p.cells-1,max(0,int(floor(((position-origin)/p.extent)*p.cells))));
}
template<bool emit>
__global__ void draw(Parameters p, const double* sx, const int64_t* sid,
                     const int32_t* cells, const double* tx, const int64_t* tid,
                     int begin, int count, uint64_t* counts, const uint64_t* offsets,
                     Edge* edges, int* mismatch) {
    int row = blockIdx.x*blockDim.x + threadIdx.x;
    if (row >= count) return;
    int t = begin + row;
    uint64_t cursor = 0;
    if (p.p0 == 0 || p.repeats == 0) { if (!emit) counts[row]=0; return; }
    curandStatePhilox4_32_10_t topology, weights;
    uint64_t stream = uint64_t(tid[t])*72 + p.stream;
    curand_init(p.seed, 2*stream, 0, &topology);
    if (emit) curand_init(p.seed, 2*stream+1, 0, &weights);
    double ax=primary_anchor(tx[2*t],p.origin_x,p);
    double ay=primary_anchor(tx[2*t+1],p.origin_y,p);
    bool extra_x=ax+p.radius-p.origin_x>p.extent;
    bool extra_y=ay+p.radius-p.origin_y>p.extent;
    double log_failure = p.p0 < 1 ? log1p(-p.p0) : 0;
    for (int r=0; r<p.repeats; ++r) for (int anchor=0; anchor<4; ++anchor) {
        if (((anchor&1) && !extra_x) || ((anchor&2) && !extra_y)) continue;
        double x=ax-((anchor&1) ? p.extent : 0);
        double y=ay-((anchor&2) ? p.extent : 0);
        int x0=grid_cell(x-p.radius,p.origin_x,p), x1=grid_cell(x+p.radius,p.origin_x,p);
        int y0=grid_cell(y-p.radius,p.origin_y,p), y1=grid_cell(y+p.radius,p.origin_y,p);
        for (int cy=y0; cy<=y1; ++cy) for (int cx=x0; cx<=x1; ++cx) {
            int c=cy*p.cells+cx;
            int index=cells[c], end=cells[c+1];
            while (index < end) {
                // Independent Bernoulli(p0) candidates by geometric skipping.
                // Each native periodic iterator entry receives a separate draw.
                if (p.p0 < 1) {
                    double skip = floor(log(curand_uniform_double(&topology))/log_failure);
                    if (skip >= double(end-index)) break;
                    index += int(skip);
                }
                int source=index++;
                if (sid[source]==tid[t]) continue;
                double dx=fabs(sx[2*source]-x), dy=fabs(sx[2*source+1]-y);
                if (dx > p.radius || dy > p.radius) continue;
                double mask_distance=sqrt(dx*dx+dy*dy);
                if (dx+dy > p.radius && mask_distance > p.radius) continue;
                // Parameter evaluation receives canonical source q and the
                // original target position, not this iterator's shifted anchor.
                dx=sx[2*source]-tx[2*t]; dy=sx[2*source+1]-tx[2*t+1];
                double inverse_extent=1./p.extent;
                dx-=p.extent*round(dx*inverse_extent);
                dy-=p.extent*round(dy*inverse_extent);
                double distance=sqrt(dx*dx+dy*dy);
                if (curand_uniform_double(&topology) > exp(-distance/p.beta)) continue;
                if (emit) {
                    if (offsets[row]+cursor >= offsets[row+1]) {
                        atomicExch(mismatch,1); return;
                    }
                    double weight=p.weight_mean;
                    if (p.weight_sd > 0) {
                        do { weight=p.weight_mean+p.weight_sd*curand_normal_double(&weights); }
                        while ((p.weight_mean < 0 && weight > 0) ||
                               (p.weight_mean >= 0 && weight < 0));
                    }
                    edges[offsets[row]+cursor] = Edge{sid[source],tid[t],weight,
                        p.delay_offset+distance/p.speed,r,anchor};
                }
                ++cursor;
            }
        }
    }
    if (emit) {
        if (offsets[row]+cursor != offsets[row+1]) atomicExch(mismatch,1);
    } else counts[row]=cursor;
}
template<class Function> int protect(char* error, size_t capacity, Function function) {
    try { function(); if (capacity) error[0]='\0'; return 0; }
    catch (const std::exception& e) { if (capacity) std::snprintf(error,capacity,"%s",e.what()); return 1; }
}
}
extern "C" int spatial_create(const double* sx, const int64_t* sid, int ns,
    const int32_t* offsets, int cells, const double* tx, const int64_t* tid, int nt,
    const double* values, int repeats, uint64_t seed, uint64_t stream,
    void** result, char* error, size_t capacity) {
    *result=nullptr;
    return protect(error,capacity,[&]() {
        auto ctx = new Context();
        try {
            ctx->p = Parameters{values[0],values[1],values[2],values[3],values[4],
                values[5],values[6],values[7],values[8],values[9],seed,stream,cells,repeats,ns,nt};
            ctx->sx.copy(sx,2*size_t(ns)); ctx->sid.copy(sid,ns);
            ctx->tx.copy(tx,2*size_t(nt)); ctx->tid.copy(tid,nt);
            ctx->cell_offsets.copy(offsets,size_t(cells)*cells+1);
        } catch (...) { delete ctx; throw; }
        *result=ctx;
    });
}
extern "C" int spatial_counts(void* handle, int begin, int count, uint64_t* host,
    char* error, size_t capacity) {
    return protect(error,capacity,[&]() {
        auto& c=*static_cast<Context*>(handle);
        if (begin<0 || count<1 || begin>c.p.target_count-count) throw std::runtime_error("invalid target slice");
        Buffer<uint64_t> counts; counts.allocate(count);
        draw<false><<<(count+127)/128,128>>>(c.p,c.sx.data,c.sid.data,c.cell_offsets.data,
            c.tx.data,c.tid.data,begin,count,counts.data,nullptr,nullptr,nullptr);
        check(cudaGetLastError()); check(cudaDeviceSynchronize());
        check(cudaMemcpy(host,counts.data,size_t(count)*sizeof(uint64_t),cudaMemcpyDeviceToHost));
    });
}
extern "C" int spatial_edges(void* handle, int begin, int count, const uint64_t* offsets,
    Edge* host, char* error, size_t capacity) {
    return protect(error,capacity,[&]() {
        auto& c=*static_cast<Context*>(handle);
        if (begin<0 || count<1 || begin>c.p.target_count-count) throw std::runtime_error("invalid target slice");
        uint64_t total=offsets[count];
        if (!total) return;
        if (total > SIZE_MAX/sizeof(Edge)) throw std::runtime_error("edge allocation overflows size_t");
        Buffer<uint64_t> device_offsets; device_offsets.copy(offsets,count+1);
        Buffer<Edge> edges; edges.allocate(total);
        int zero=0, mismatch=0;
        Buffer<int> flag; flag.copy(&zero,1);
        draw<true><<<(count+127)/128,128>>>(c.p,c.sx.data,c.sid.data,c.cell_offsets.data,
            c.tx.data,c.tid.data,begin,count,nullptr,device_offsets.data,edges.data,flag.data);
        check(cudaGetLastError()); check(cudaDeviceSynchronize());
        check(cudaMemcpy(&mismatch,flag.data,sizeof(int),cudaMemcpyDeviceToHost));
        if (mismatch) throw std::runtime_error("count and emission passes disagree");
        check(cudaMemcpy(host,edges.data,size_t(total)*sizeof(Edge),cudaMemcpyDeviceToHost));
    });
}
extern "C" void spatial_destroy(void* handle) { delete static_cast<Context*>(handle); }

namespace {
__global__ void initialize_layout(const int64_t* ids, const double* means, const double* deviations,
    int count, double extent, uint64_t layout_seed, uint64_t state_seed, double* xy, double* vm) {
    int index=blockIdx.x*blockDim.x+threadIdx.x;
    if (index>=count) return;
    curandStatePhilox4_32_10_t spatial,state;
    curand_init(layout_seed,uint64_t(ids[index]),0,&spatial);
    curand_init(state_seed,uint64_t(ids[index]),0,&state);
    // cuRAND U is in (0,1]; this maps to the half-open source square [-L/2,L/2).
    xy[2*index]=extent*(.5-curand_uniform_double(&spatial));
    xy[2*index+1]=extent*(.5-curand_uniform_double(&spatial));
    vm[index]=means[index]+deviations[index]*curand_normal_double(&state);
}
}
extern "C" int spatial_layout(const int64_t* ids, const double* means, const double* deviations,
    int count, double extent, uint64_t layout_seed, uint64_t state_seed, double* xy, double* vm,
    char* error, size_t capacity) {
    return protect(error,capacity,[&]() {
        if (count<1 || !std::isfinite(extent) || extent<=0) throw std::runtime_error("invalid layout");
        Buffer<int64_t> did; did.copy(ids,count);
        Buffer<double> dmean,dsd,dxy,dvm;
        dmean.copy(means,count); dsd.copy(deviations,count);
        dxy.allocate(size_t(count)*2); dvm.allocate(count);
        initialize_layout<<<(count+127)/128,128>>>(did.data,dmean.data,dsd.data,count,extent,
            layout_seed,state_seed,dxy.data,dvm.data);
        check(cudaGetLastError()); check(cudaDeviceSynchronize());
        check(cudaMemcpy(xy,dxy.data,size_t(count)*2*sizeof(double),cudaMemcpyDeviceToHost));
        check(cudaMemcpy(vm,dvm.data,size_t(count)*sizeof(double),cudaMemcpyDeviceToHost));
    });
}
