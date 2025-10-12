//
// Created by auser on 5/24/24.
//

#include "PointLight.h"

PointLight::PointLight( const Vec3d& _origin, double _intensity ): origin( _origin ) {
    intensity = _intensity;
}

PointLight::PointLight( const Vec3d& _origin, double _intensity, const RGB& _lightColor ): origin( _origin ) {
    intensity = _intensity;
    lightColor = _lightColor;
}

Light::Type PointLight::getType() const {
    return POINT;
}

bool PointLight::isIntersectsWithRay( const Ray& ray ) const {
    return false;
}

Vec3d PointLight::getSamplePoint() const {
    return origin;
}
