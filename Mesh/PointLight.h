//
// Created by auser on 5/24/24.
//

#ifndef COLLECTION_POINTLIGHT_H
#define COLLECTION_POINTLIGHT_H
#include "Light.h"
#include "Color.h"

class PointLight: public Light {
public:
    __host__ __device__ PointLight( const Vector3f& _origin, float _intensity );

    __host__ __device__ PointLight( const Vector3f& _origin, float _intensity, const RGB& _lightColor );

    __host__ __device__ Type getType() const override;

    __host__ __device__ bool isIntersectsWithRay( const Ray& ray ) const override;

    __device__ Vector3f getSamplePoint() const override;
private:
    Vector3f origin;
};


#endif //COLLECTION_POINTLIGHT_H
