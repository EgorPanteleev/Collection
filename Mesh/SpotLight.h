//
// Created by auser on 5/24/24.
//

#ifndef COLLECTION_SPOTLIGHT_H
#define COLLECTION_SPOTLIGHT_H
#include "Light.h"

#include "Color.h"

class SpotLight: public Light {
public:
    SpotLight( const Vector3f& _pMin, const Vector3f& _pMax, float _intensity );

    SpotLight( const Vector3f& _pMin, const Vector3f& _pMax, float _intensity, const RGB& _lightColor );

    Type getType() const override;

    bool isIntersectsWithRay( const Ray& ray ) const override;

    Vector3f getSamplePoint() const override;
private:
    Vector3f pMin, pMax;
};

#endif //COLLECTION_SPOTLIGHT_H
