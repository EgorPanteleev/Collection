//
// Created by auser on 5/24/24.
//

#include "SpotLight.h"
#include <random>
#include <hiprand/hiprand_kernel.h>

__host__ __device__ SpotLight::SpotLight( const Vector3f& _pMin, const Vector3f& _pMax, float _intensity ): pMin( _pMin ), pMax( _pMax ) {
    intensity = _intensity;
}

__host__ __device__ SpotLight::SpotLight( const Vector3f& _pMin, const Vector3f& _pMax, float _intensity, const RGB& _lightColor ): pMin( _pMin ), pMax( _pMax ) {
    intensity = _intensity;
    lightColor = _lightColor;
}

__host__ __device__ Light::Type SpotLight::getType() const {
    return SPOT;
}


__host__ __device__ bool SpotLight::isIntersectsWithRay( const Ray& ray ) const {
    float tx1 = (pMin.x - ray.origin.x) / ray.direction.x, tx2 = (pMax.x - ray.origin.x) / ray.direction.x;
    float tmin = std::min( tx1, tx2 ), tmax = std::max( tx1, tx2 );
    float ty1 = (pMin.y - ray.origin.y) / ray.direction.y, ty2 = (pMax.y - ray.origin.y) / ray.direction.y;
    tmin = std::max( tmin, std::min( ty1, ty2 ) ), tmax = std::min( tmax, std::max( ty1, ty2 ) );
    float tz1 = (pMin.z - ray.origin.z) / ray.direction.z, tz2 = (pMax.z - ray.origin.z) / ray.direction.z;
    tmin = std::max( tmin, std::min( tz1, tz2 ) ), tmax = std::min( tmax, std::max( tz1, tz2 ) );
    return tmax >= tmin && tmin < 1e30f && tmax > 0;
}

__device__ Vector3f SpotLight::getSamplePoint() const {
    hiprandState state;
    hiprand_init( 1, 1, 0, &state );
    float randX = pMin.x + ( hiprand_uniform(&state) ) * ( pMax.x - pMin.x );
    float randZ = pMin.z + ( hiprand_uniform(&state) ) * ( pMax.z - pMin.z );
    return { randX, ( pMax.y + pMin.y ) / 2, randZ };
}
