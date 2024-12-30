//
// Created by auser on 11/26/24.
//

#ifdef HIP_ENABLED

#define GLOBAL __global__
#define HOST __host__
#define DEVICE __device__
#define HOST_DEVICE __host__ __device__

#ifdef NDEBUG
#define HIP_ASSERT(x) x
#else
#define HIP_ASSERT(x) (assert((x)==hipSuccess))
#endif

#pragma once
#include "hip/hip_runtime.h"

namespace HIP {

    template<typename Type>
    [[nodiscard]] Type* allocateOnDevice( size_t n = 1 ) {
        Type* device;
        HIP_ASSERT(hipMalloc(&device, n * sizeof(Type)));
        return device;
    }

    template<typename Type>
    void allocateOnDevice( Type*& device, size_t n = 1 ) {
        HIP_ASSERT(hipMalloc(&device, n * sizeof(Type)));
    }

    template<typename Type>
    void copyToDevice( Type* host, Type* device, size_t n = 1 ) {
        HIP_ASSERT(hipMemcpy(device, host, n * sizeof(Type), hipMemcpyHostToDevice));
    }

    template<typename Type>
    void copyToHost(Type* host, Type* device, size_t n = 1) {
        HIP_ASSERT(hipMemcpy(host, device, n * sizeof(Type), hipMemcpyDeviceToHost));
    }

    template<typename Type>
    void deallocateOnDevice( Type* device ) {
        HIP_ASSERT(hipFree(device));
    }

}

#else

#define GLOBAL
#define HOST
#define DEVICE
#define HOST_DEVICE

#endif

