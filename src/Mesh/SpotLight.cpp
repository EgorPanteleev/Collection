//
// Created by auser on 5/24/24.
//

#include "SpotLight.h"
#include "Random.h"

SpotLight::SpotLight( const Vec3d& _pMin, const Vec3d& _pMax, double _intensity ): pMin( _pMin ), pMax( _pMax ) {
    intensity = _intensity;
}

SpotLight::SpotLight( const Vec3d& _pMin, const Vec3d& _pMax, double _intensity, const RGB& _lightColor ): pMin( _pMin ), pMax( _pMax ) {
    intensity = _intensity;
    lightColor = _lightColor;
}

Light::Type SpotLight::getType() const {
    return SPOT;
}


bool SpotLight::isIntersectsWithRay( const Ray& ray ) const {
    double tx1 = (pMin[0] - ray.origin[0]) / ray.direction[0], tx2 = (pMax[0] - ray.origin[0]) / ray.direction[0];
    double tmin = std::min( tx1, tx2 ), tmax = std::max( tx1, tx2 );
    double ty1 = (pMin[1] - ray.origin[1]) / ray.direction[1], ty2 = (pMax[1] - ray.origin[1]) / ray.direction[1];
    tmin = std::max( tmin, std::min( ty1, ty2 ) ), tmax = std::min( tmax, std::max( ty1, ty2 ) );
    double tz1 = (pMin[2] - ray.origin[2]) / ray.direction[2], tz2 = (pMax[2] - ray.origin[2]) / ray.direction[2];
    tmin = std::max( tmin, std::min( tz1, tz2 ) ), tmax = std::min( tmax, std::max( tz1, tz2 ) );
    return tmax >= tmin && tmin < 1e30 && tmax > 0;
}

Vec3d SpotLight::getSamplePoint() const {
    double randX = pMin[0] + randomDouble() * ( pMax[0] - pMin[0] );
    double randZ = pMin[2] + randomDouble() * ( pMax[2] - pMin[2] );
    return { randX, ( pMax[1] + pMin[1] ) / 2, randZ };
}
