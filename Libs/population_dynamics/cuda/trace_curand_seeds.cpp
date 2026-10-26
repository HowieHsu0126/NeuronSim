// Diagnostic preload: retain exactly the original curandGenerate call/results,
// and record scalar seeds passed from the host generator to per-edge streams.
// Loading this shim alone does not constitute a validated source intervention.
#include <cuda_runtime_api.h>
#include <curand.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>

extern "C" curandStatus_t curandGenerate(curandGenerator_t generator,
                                        unsigned int *output, size_t count) {
    using Function = curandStatus_t (*)(curandGenerator_t,unsigned int*,size_t);
    static auto original = reinterpret_cast<Function>(dlsym(RTLD_NEXT,"curandGenerate"));
    if (!original) return CURAND_STATUS_INTERNAL_ERROR;
    const auto status = original(generator,output,count);
    if (status != CURAND_STATUS_SUCCESS || count != 1) return status;
    const char *path = std::getenv("NEURONSIM_CURAND_SEED_TRACE");
    if (!path) return CURAND_STATUS_SUCCESS;
    unsigned int value=0;
    if (cudaMemcpy(&value,output,sizeof(value),cudaMemcpyDeviceToHost) != cudaSuccess)
        return CURAND_STATUS_INTERNAL_ERROR;
    FILE *stream=std::fopen(path,"a");
    if (!stream) return CURAND_STATUS_INTERNAL_ERROR;
    const bool wrote=std::fprintf(stream,"%u\n",value)>0;
    const int closed=std::fclose(stream);
    return wrote && closed==0 ? CURAND_STATUS_SUCCESS : CURAND_STATUS_INTERNAL_ERROR;
}
