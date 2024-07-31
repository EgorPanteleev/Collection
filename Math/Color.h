
#ifndef COLLECTION_COLOR_H
#define COLLECTION_COLOR_H
#include "hip/hip_runtime.h"
#undef RGB
class RGB {
public:
    __host__ __device__ void set( float _r, float _g, float _b );
    __host__ __device__ RGB operator+( const RGB& color ) const;
    __host__ __device__ RGB operator*( float a) const;
    __host__ __device__ RGB operator/( float a) const;
    __host__ __device__ bool operator==( const RGB& color ) const;
    __host__ __device__ void scaleTo( float value );
    __host__ __device__ RGB();
    __host__ __device__ RGB( float _r, float _g, float _b);
    __host__ __device__ ~RGB();
public:
    float r;
    float g;
    float b;
};


#endif //COLLECTION_COLOR_H
