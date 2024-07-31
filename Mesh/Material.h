#pragma once
#include "Color.h"
class Material {
public:
    __host__ __device__ Material();
    __host__ __device__ Material( const RGB& color, float diffuse, float reflection );
    [[nodiscard]] __host__ __device__ RGB getColor() const;
    __host__ __device__ void setColor( const RGB& c );
    [[nodiscard]] __host__ __device__ float getDiffuse() const;
    __host__ __device__ void setDiffuse( float d );
    [[nodiscard]] __host__ __device__ float getReflection() const;
    __host__ __device__ void setReflection( float r );
private:
    RGB color;
    float diffuse;
    float reflection;
};

