// Standalone cuRAND Poisson streams: no neuronal dynamics or event buffers.
// Binary output contains per-tick totals and selected target counts, uint64.
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>

static void checked(cudaError_t result) {
    if(result!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}

template<class State> __global__ void generate(unsigned int seed,int count,int ticks,float mean,
                                              int skip_zero_calls,int sampled_count,const int* selected,
                                              unsigned long long* totals,unsigned long long* samples) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=count) return;
    State state; curand_init(seed,i,0,&state);
    for(int t=0;t<skip_zero_calls;++t) curand_poisson(&state,0.0f);
    int sample_index=-1;
    for(int k=0;k<sampled_count;++k) if(selected[k]==i) sample_index=k;
    for(int t=0;t<ticks;++t) {
        const unsigned int n=curand_poisson(&state,mean);
        atomicAdd(totals+t,static_cast<unsigned long long>(n));
        if(sample_index>=0) samples[t*sampled_count+sample_index]=n;
    }
}

int main(int argc,char** argv) {
    // seeds/input file: seed count rate_hz selected_count selected indices ...
    if(argc!=8) { std::fprintf(stderr,"usage: input output ticks dt_ms skip_zero_calls xorwow|philox groups\n");return 2; }
    try {
        FILE* input=std::fopen(argv[1],"r");if(!input)throw std::runtime_error("input open failed");
        FILE* output=std::fopen(argv[2],"wb");if(!output)throw std::runtime_error("output open failed");
        const int ticks=std::atoi(argv[3]),skip=std::atoi(argv[5]),groups=std::atoi(argv[7]);
        const float dt=std::strtof(argv[4],nullptr);
        if(ticks<=0 || skip<0 || groups<=0 || dt<=0) throw std::runtime_error("invalid dimensions");
        for(int group=0;group<groups;++group) {
            unsigned int seed;int count,k;float rate;
            if(std::fscanf(input,"%u %d %f %d",&seed,&count,&rate,&k)!=4 || count<1 || k<1 || rate<0)
                throw std::runtime_error("invalid group input");
            std::vector<int> selected(k);
            for(int& i:selected) if(std::fscanf(input,"%d",&i)!=1 || i<0 || i>=count)
                throw std::runtime_error("invalid selected index");
            int* d_selected;unsigned long long *d_total,*d_sample;
            checked(cudaMalloc(&d_selected,k*sizeof(int)));
            checked(cudaMemcpy(d_selected,selected.data(),k*sizeof(int),cudaMemcpyHostToDevice));
            checked(cudaMalloc(&d_total,ticks*sizeof(unsigned long long)));
            checked(cudaMemset(d_total,0,ticks*sizeof(unsigned long long)));
            checked(cudaMalloc(&d_sample,static_cast<size_t>(ticks)*k*sizeof(unsigned long long)));
            const float mean=dt*rate/1000.0f;
            if(std::string(argv[6])=="xorwow")
                generate<curandState><<<(count+255)/256,256>>>(seed,count,ticks,mean,skip,k,d_selected,d_total,d_sample);
            else if(std::string(argv[6])=="philox")
                generate<curandStatePhilox4_32_10_t><<<(count+255)/256,256>>>(seed,count,ticks,mean,skip,k,d_selected,d_total,d_sample);
            else throw std::runtime_error("unknown generator");
            checked(cudaGetLastError());checked(cudaDeviceSynchronize());
            std::vector<unsigned long long> totals(ticks),samples(static_cast<size_t>(ticks)*k);
            checked(cudaMemcpy(totals.data(),d_total,totals.size()*sizeof(unsigned long long),cudaMemcpyDeviceToHost));
            checked(cudaMemcpy(samples.data(),d_sample,samples.size()*sizeof(unsigned long long),cudaMemcpyDeviceToHost));
            if(std::fwrite(totals.data(),sizeof(unsigned long long),totals.size(),output)!=totals.size() ||
               std::fwrite(samples.data(),sizeof(unsigned long long),samples.size(),output)!=samples.size())
                throw std::runtime_error("incomplete output");
            checked(cudaFree(d_selected));checked(cudaFree(d_total));checked(cudaFree(d_sample));
        }
        if(std::fclose(input)!=0 || std::fclose(output)!=0) throw std::runtime_error("file close failed");
        return 0;
    } catch(const std::exception& error) { std::fprintf(stderr,"%s\n",error.what());return 1; }
}
