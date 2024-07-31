//
// Created by auser on 5/24/24.
//

#ifndef COLLECTION_SPOTLIGHT_H
#define COLLECTION_SPOTLIGHT_H
#include "Light.h"

#include "Color.h"

class SpotLight: public Light {
public:
    __host__ __device__ SpotLight( const Vector3f& _pMin, const Vector3f& _pMax, float _intensity );

    __host__ __device__ SpotLight( const Vector3f& _pMin, const Vector3f& _pMax, float _intensity, const RGB& _lightColor );

    __host__ __device__ Type getType() const override;

    __host__ __device__ bool isIntersectsWithRay( const Ray& ray ) const override;

    __device__ Vector3f getSamplePoint() const override;
private:
    Vector3f pMin, pMax;
};

#endif //COLLECTION_SPOTLIGHT_H
