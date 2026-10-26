// Streaming weak connectivity: monotone atomic union and path splitting.
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <stdexcept>

namespace {
void check(cudaError_t e) { if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
struct Components { int count; int* parent=nullptr; ~Components(){if(parent) cudaFree(parent);} };
__global__ void initialize(int* parent,int n) { int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)parent[i]=i; }
__device__ int root(int* parent,int node) {
    while(node!=0) {
        int p=atomicAdd(parent+node,0);
        if(p==node) return node;
        int grand=atomicAdd(parent+p,0);
        atomicCAS(parent+node,p,grand);
        node=p;
    }
    return 0;  // Monotone unions can never move the smallest vertex.
}
__global__ void unite(int* parent,const int* source,const int* target,int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n)return;
    int a=source[i],b=target[i];
    while(true) {
        a=root(parent,a);b=root(parent,b);
        if(a==b)return;
        int high=max(a,b),low=min(a,b);
        if(atomicCAS(parent+high,high,low)==high)return;
    }
}
__global__ void flatten(int* parent,int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n)atomicExch(parent+i,root(parent,i));
}
template<class F> int protect(char* error,size_t capacity,F f) {
    try {f();if(capacity)error[0]='\0';return 0;}
    catch(const std::exception& e){if(capacity)std::snprintf(error,capacity,"%s",e.what());return 1;}
}
}
extern "C" int components_create(int n,void** output,char* error,size_t capacity) {
    *output=nullptr;
    return protect(error,capacity,[&](){
        if(n<1)throw std::runtime_error("positive vertex count required");
        auto c=new Components();c->count=n;
        try {
            check(cudaMalloc(reinterpret_cast<void**>(&c->parent),size_t(n)*sizeof(int)));
            initialize<<<(n+255)/256,256>>>(c->parent,n);
            check(cudaGetLastError());check(cudaDeviceSynchronize());
        }catch(...){delete c;throw;}
        *output=c;
    });
}
extern "C" int components_add(void* handle,const int* source,const int* target,int n,char* error,size_t capacity) {
    return protect(error,capacity,[&](){
        if(n<0)throw std::runtime_error("negative edge count");
        if(!n)return;
        auto& c=*static_cast<Components*>(handle);
        int* edges=nullptr;
        try {
            check(cudaMalloc(reinterpret_cast<void**>(&edges),2*size_t(n)*sizeof(int)));
            check(cudaMemcpy(edges,source,size_t(n)*sizeof(int),cudaMemcpyHostToDevice));
            check(cudaMemcpy(edges+n,target,size_t(n)*sizeof(int),cudaMemcpyHostToDevice));
            unite<<<(n+255)/256,256>>>(c.parent,edges,edges+n,n);
            check(cudaGetLastError());check(cudaDeviceSynchronize());
            check(cudaFree(edges));edges=nullptr;
        }catch(...){if(edges)cudaFree(edges);throw;}
    });
}
extern "C" int components_labels(void* handle,int* labels,char* error,size_t capacity) {
    return protect(error,capacity,[&](){
        auto& c=*static_cast<Components*>(handle);
        flatten<<<(c.count+255)/256,256>>>(c.parent,c.count);
        check(cudaGetLastError());check(cudaDeviceSynchronize());
        check(cudaMemcpy(labels,c.parent,size_t(c.count)*sizeof(int),cudaMemcpyDeviceToHost));
    });
}
extern "C" void components_destroy(void* handle){delete static_cast<Components*>(handle);}
