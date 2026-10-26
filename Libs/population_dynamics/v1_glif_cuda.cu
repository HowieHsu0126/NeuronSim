// SPDX-License-Identifier: GPL-2.0-or-later
// GLIF update translation: Copyright (C) 2004 The NEST Initiative.
// Adapted from pinned NEST glif_psc; see ../external/nest_glif_reference/LICENSE.
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace {
void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
template <typename T> struct DeviceArray {
    T* data = nullptr;
    size_t count;
    explicit DeviceArray(size_t n) : count(n) { check(cudaMalloc(&data, n * sizeof(T))); }
    ~DeviceArray() { if (data) cudaFree(data); }
    void from(const T* source) { check(cudaMemcpy(data, source, count * sizeof(T), cudaMemcpyHostToDevice)); }
    void to(T* target) const { check(cudaMemcpy(target, data, count * sizeof(T), cudaMemcpyDeviceToHost)); }
    void zero() { check(cudaMemset(data, 0, count * sizeof(T))); }
};
struct Event {
    cudaEvent_t event;
    Event() { check(cudaEventCreate(&event)); }
    ~Event() { cudaEventDestroy(event); }
};
enum Scalar { EL, THRESHOLD, RESET, P33, P30, ASC_AMP0, ASC_AMP1, ASC_DECAY0,
              ASC_DECAY1, ASC_MEAN0, ASC_MEAN1, ASC_REF0, ASC_REF1, WEIGHT, SCALARS };
enum Synapse { P11, P21, P31, P32, PSC_INITIAL, SYNAPSES };
// State rows: relative voltage, ASC0, ASC1, ASC sum, recorded synaptic current,
// followed by R first and R second alpha-PSC states.
__global__ void advance(int n, int r, int steps, int step, double dt,
                        const double* scalar, const double* synapse,
                        const int* integer, const int* drive,
                        double* state, int* refractory, double* signals,
                        unsigned char* spikes, double* ring, int slots,
                        int* active_count, int* active) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double u = state[i];
    double a0 = state[n + i], a1 = state[2*n + i];
    double asc_sum = state[3*n + i], isyn = state[4*n + i];
    bool fired = false;
    if (refractory[i] == 0) {
        asc_sum = scalar[ASC_MEAN0*n+i]*a0 + scalar[ASC_MEAN1*n+i]*a1;
        a0 *= scalar[ASC_DECAY0*n+i];
        a1 *= scalar[ASC_DECAY1*n+i];
        u = u*scalar[P33*n+i] + asc_sum*scalar[P30*n+i];
        isyn = 0.0;
        for (int receptor = 0; receptor < r; ++receptor) {
            const double y1 = state[(5+receptor)*n+i];
            const double y2 = state[(5+r+receptor)*n+i];
            u += synapse[(P31*r+receptor)*n+i]*y1
                 + synapse[(P32*r+receptor)*n+i]*y2;
            isyn += y2;
        }
        if (u > scalar[THRESHOLD*n+i]) {
            refractory[i] = integer[i];
            a0 = scalar[ASC_AMP0*n+i] + a0*scalar[ASC_REF0*n+i];
            a1 = scalar[ASC_AMP1*n+i] + a1*scalar[ASC_REF1*n+i];
            u = scalar[RESET*n+i];
            fired = true;
        }
    } else {
        --refractory[i];
    }
    const int input_receptor = integer[2*n+i];
    const int input_count = ring ? 0 : drive[integer[n+i]*steps+step];
    for (int receptor = 0; receptor < r; ++receptor) {
        double y1 = state[(5+receptor)*n+i];
        double y2 = state[(5+r+receptor)*n+i];
        y2 = synapse[(P21*r+receptor)*n+i]*y1
             + synapse[(P11*r+receptor)*n+i]*y2;
        y1 *= synapse[(P11*r+receptor)*n+i];
        if (ring) {
            const size_t index=(static_cast<size_t>(step%slots)*r+receptor)*n+i;
            y1 += synapse[(PSC_INITIAL*r+receptor)*n+i]*ring[index];
            ring[index]=0.0;
        } else if (receptor == input_receptor) {
            y1 += synapse[(PSC_INITIAL*r+receptor)*n+i]*scalar[WEIGHT*n+i]*input_count;
        }
        state[(5+receptor)*n+i] = y1;
        state[(5+r+receptor)*n+i] = y2;
    }
    state[i] = u; state[n+i] = a0; state[2*n+i] = a1;
    state[3*n+i] = asc_sum; state[4*n+i] = isyn;
    spikes[static_cast<size_t>(i)*steps+step] = fired;
    if (fired && active_count) active[atomicAdd(active_count,1)]=i;
    if (signals && step < steps-1) {
        const double values[7] = {u+scalar[EL*n+i], isyn, 0.0, asc_sum,
                                 scalar[THRESHOLD*n+i]+scalar[EL*n+i], 0.0, 0.0};
        for (int signal=0; signal<7; ++signal)
            signals[(static_cast<size_t>(signal)*n+i)*(steps-1)+step] = values[signal];
    }
}
using Clock = std::chrono::steady_clock;
double milliseconds(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end-start).count();
}
} // namespace

extern "C" int glif_device_info(int* values, char* name, int name_size,
                                char* error, int error_size) {
    try {
        int count; check(cudaGetDeviceCount(&count));
        if (count != 1) throw std::runtime_error("Exactly one scheduler-visible GPU required");
        check(cudaSetDevice(0));
        cudaDeviceProp props; check(cudaGetDeviceProperties(&props, 0));
        values[0]=props.major; values[1]=props.minor;
        check(cudaRuntimeGetVersion(&values[2])); check(cudaDriverGetVersion(&values[3]));
        std::snprintf(name,name_size,"%s",props.name);
        return 0;
    } catch (const std::exception& e) { std::snprintf(error,error_size,"%s",e.what()); return 1; }
}

extern "C" int glif_component_run(int n, int r, int steps, int protocols, double dt,
                                  const double* scalar, const double* synapse,
                                  const int* integer, const int* drive,
                                  const double* initial_state, double* signals,
                                  double* terminal_state, unsigned char* spikes,
                                  double* metrics, char* error, int error_size) {
    try {
        if (n<1 || r<1 || steps<2 || protocols<1 || dt<=0)
            throw std::runtime_error("Invalid GLIF component dimensions");
        const auto started=Clock::now();
        const size_t states=static_cast<size_t>(5+2*r)*n;
        DeviceArray<double> ds(static_cast<size_t>(SCALARS)*n);
        DeviceArray<double> dy(static_cast<size_t>(SYNAPSES)*r*n);
        DeviceArray<int> di(3UL*n), dd(static_cast<size_t>(protocols)*steps), dr(n);
        DeviceArray<double> state(states), output(7UL*n*(steps-1));
        DeviceArray<unsigned char> fired(static_cast<size_t>(n)*steps);
        ds.from(scalar); dy.from(synapse); di.from(integer); dd.from(drive);
        state.from(initial_state); dr.zero();
        const auto ready=Clock::now();
        Event first,last;
        check(cudaEventRecord(first.event));
        for (int step=0; step<steps; ++step) {
            advance<<<(n+127)/128,128>>>(n,r,steps,step,dt,ds.data,dy.data,di.data,
                dd.data,state.data,dr.data,output.data,fired.data,nullptr,0,nullptr,nullptr);
            check(cudaGetLastError());
        }
        check(cudaEventRecord(last.event)); check(cudaEventSynchronize(last.event));
        float kernel_ms; check(cudaEventElapsedTime(&kernel_ms,first.event,last.event));
        const auto complete=Clock::now();
        output.to(signals); state.to(terminal_state); fired.to(spikes);
        const auto copied=Clock::now();
        metrics[0]=kernel_ms; metrics[1]=milliseconds(started,ready);
        metrics[2]=milliseconds(complete,copied); metrics[3]=milliseconds(started,copied);
        metrics[4]=(ds.count+dy.count+state.count+output.count)*sizeof(double)
                   +(di.count+dd.count+dr.count)*sizeof(int)+fired.count;
        return 0;
    } catch (const std::exception& e) { std::snprintf(error,error_size,"%s",e.what()); return 1; }
}

namespace {
__global__ void propagate_events(int n, int r, int slots, int step,
    const long long* offsets, const int* targets, const int* receptors,
    const int* delays, const double* weights, const int* active,
    const int* active_count, const long long* event_offsets, double* ring) {
    const long long begin = active_count ? 0 : event_offsets[step];
    const long long end = active_count ? *active_count : event_offsets[step+1];
    for (long long event=begin+blockIdx.x; event<end; event+=gridDim.x) {
        const int source=active[event];
        for (long long edge=offsets[source]+threadIdx.x; edge<offsets[source+1]; edge+=blockDim.x) {
            const size_t slot=(step+delays[edge])%slots;
            atomicAdd(ring+(slot*r+receptors[edge])*n+targets[edge],weights[edge]);
        }
    }
}
} // namespace

extern "C" int glif_network_run(int n,int r,int steps,int sources,int edges,int events,
    int slots,double dt,const double* scalar,const double* synapse,const int* integer,
    const double* initial_state,const long long* offsets,const int* targets,
    const int* receptors,const int* delays,const double* weights,
    const long long* event_offsets,const int* event_sources,
    double* terminal_state,unsigned char* spikes,double* metrics,char* error,int error_size) {
    try {
        if(n<1 || r<1 || steps<2 || sources<n || edges<1 || events<1 || slots<2 || dt<=0)
            throw std::runtime_error("Invalid GLIF network dimensions");
        const auto started=Clock::now();
        DeviceArray<double> ds(static_cast<size_t>(SCALARS)*n),dy(static_cast<size_t>(SYNAPSES)*r*n);
        DeviceArray<int> di(3UL*n),dr(n),active(n),active_count(1);
        DeviceArray<double> state(static_cast<size_t>(5+2*r)*n),ring(static_cast<size_t>(slots)*r*n);
        DeviceArray<unsigned char> fired(static_cast<size_t>(n)*steps);
        DeviceArray<long long> doff(static_cast<size_t>(sources)+1),deoff(static_cast<size_t>(steps)+1);
        DeviceArray<int> target(edges),receptor(edges),delay(edges),external(events);
        DeviceArray<double> weight(edges);
        ds.from(scalar);dy.from(synapse);di.from(integer);state.from(initial_state);
        doff.from(offsets);target.from(targets);receptor.from(receptors);delay.from(delays);
        weight.from(weights);deoff.from(event_offsets);external.from(event_sources);
        dr.zero();ring.zero();
        const auto ready=Clock::now();
        Event first,last;check(cudaEventRecord(first.event));
        for(int step=0;step<steps;++step){
            check(cudaMemsetAsync(active_count.data,0,sizeof(int)));
            advance<<<(n+127)/128,128>>>(n,r,steps,step,dt,ds.data,dy.data,di.data,nullptr,
                state.data,dr.data,nullptr,fired.data,ring.data,slots,active_count.data,active.data);
            check(cudaGetLastError());
            propagate_events<<<128,128>>>(n,r,slots,step,doff.data,target.data,receptor.data,
                delay.data,weight.data,active.data,active_count.data,nullptr,ring.data);
            check(cudaGetLastError());
            propagate_events<<<128,128>>>(n,r,slots,step,doff.data,target.data,receptor.data,
                delay.data,weight.data,external.data,nullptr,deoff.data,ring.data);
            check(cudaGetLastError());
        }
        check(cudaEventRecord(last.event));check(cudaEventSynchronize(last.event));
        float kernel_ms;check(cudaEventElapsedTime(&kernel_ms,first.event,last.event));
        const auto complete=Clock::now();
        state.to(terminal_state);fired.to(spikes);const auto copied=Clock::now();
        metrics[0]=kernel_ms;metrics[1]=milliseconds(started,ready);
        metrics[2]=milliseconds(complete,copied);metrics[3]=milliseconds(started,copied);
        metrics[4]=(ds.count+dy.count+state.count+ring.count+weight.count)*sizeof(double)
            +(di.count+dr.count+active.count+active_count.count+target.count+receptor.count+delay.count+external.count)*sizeof(int)
            +(doff.count+deoff.count)*sizeof(long long)+fired.count;
        return 0;
    }catch(const std::exception& e){std::snprintf(error,error_size,"%s",e.what());return 1;}
}
