//
// Created by auser on 5/24/24.
//

#ifndef COLLECTION_SPOTLIGHT_H
#define COLLECTION_SPOTLIGHT_H
#include "Light.h"

#include "RGB.h"

class SpotLight: public Light {
public:
    SpotLight( const Vec3d& _pMin, const Vec3d& _pMax, double _intensity );

    SpotLight( const Vec3d& _pMin, const Vec3d& _pMax, double _intensity, const RGB& _lightColor );

    Type getType() const override;

    bool isIntersectsWithRay( const Ray& ray ) const override;

    Vec3d getSamplePoint() const override;
private:
    Vec3d pMin, pMax;
};

#endif //COLLECTION_SPOTLIGHT_H
