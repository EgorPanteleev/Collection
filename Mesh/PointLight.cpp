//
// Created by auser on 5/24/24.
//

#include "PointLight.h"

PointLight::PointLight( const Vector3f& _origin, float _intensity ): origin( _origin ) {
    intensity = _intensity;
}

PointLight::PointLight( const Vector3f& _origin, float _intensity, const RGB& _lightColor ): origin( _origin ) {
    intensity = _intensity;
    lightColor = _lightColor;
}

Light::Type PointLight::getType() const {
    return POINT;
}

bool PointLight::isIntersectsWithRay( const Ray& ray ) const {
    return false;
}

Vector3f PointLight::getSamplePoint() const {
    return origin;
}
