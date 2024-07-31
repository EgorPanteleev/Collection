//
// Created by auser on 5/24/24.
//

#include "PointLight.h"

__host__ __device__ PointLight::PointLight( const Vector3f& _origin, float _intensity ): origin( _origin ) {
    intensity = _intensity;
}

__host__ __device__ PointLight::PointLight( const Vector3f& _origin, float _intensity, const RGB& _lightColor ): origin( _origin ) {
    intensity = _intensity;
    lightColor = _lightColor;
}

__host__ __device__ Light::Type PointLight::getType() const {
    return POINT;
}

__host__ __device__ bool PointLight::isIntersectsWithRay( const Ray& ray ) const {
    return false;
}

__device__ Vector3f PointLight::getSamplePoint() const {
    return origin;
}
