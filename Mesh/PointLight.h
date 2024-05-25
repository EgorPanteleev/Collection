//
// Created by auser on 5/24/24.
//

#ifndef COLLECTION_POINTLIGHT_H
#define COLLECTION_POINTLIGHT_H
#include "Light.h"
#include "Color.h"

class PointLight: public Light {
public:
    PointLight( const Vector3f& _origin, float _intensity );

    PointLight( const Vector3f& _origin, float _intensity, const RGB& _lightColor );

    Type getType() const override;

    bool isIntersectsWithRay( const Ray& ray ) const override;

    Vector3f getSamplePoint() const override;
private:
    Vector3f origin;
};


#endif //COLLECTION_POINTLIGHT_H
