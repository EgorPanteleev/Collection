//
// Created by auser on 11/26/24.
//

#ifdef HIP_ENABLED

#pragma once
#include "hip/hip_runtime.h"
#include "iostream"

#define GLOBAL __global__
#define HOST __host__
#define DEVICE __device__
#define HOST_DEVICE __host__ __device__

#define HIP_CHECK(cmd) \
{ \
    hipError_t error = cmd; \
    if (error != hipSuccess) { \
        std::cerr << "HIP error: " << hipGetErrorString(error) << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

namespace HIP {

    template<typename Type>
    [[nodiscard]] Type* allocateOnDevice( size_t n = 1 ) {
        Type* device;
        HIP_CHECK(hipMalloc(&device, n * sizeof(Type)));
        return device;
    }

    template<typename Type>
    void allocateOnDevice( Type*& device, size_t n = 1 ) {
        HIP_CHECK(hipMalloc(&device, n * sizeof(Type)));
    }

    template<typename Type>
    void copyToDevice( Type* host, Type* device, size_t n = 1 ) {
        HIP_CHECK(hipMemcpy(device, host, n * sizeof(Type), hipMemcpyHostToDevice));
    }

    template<typename Type>
    void copyToHost(Type* host, Type* device, size_t n = 1) {
        HIP_CHECK(hipMemcpy(host, device, n * sizeof(Type), hipMemcpyDeviceToHost));
    }

    template<typename Type>
    void deallocateOnDevice( Type* device ) {
        HIP_CHECK(hipFree(device));
    }

}

#else

#define GLOBAL
#define HOST
#define DEVICE
#define HOST_DEVICE

#endif

