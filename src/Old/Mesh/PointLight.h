//
// Created by auser on 5/24/24.
//

#ifndef COLLECTION_POINTLIGHT_H
#define COLLECTION_POINTLIGHT_H
#include "Light.h"
#include "RGB.h"

class PointLight: public Light {
public:
    PointLight( const Vec3d& _origin, double _intensity );

    PointLight( const Vec3d& _origin, double _intensity, const RGB& _lightColor );

    Type getType() const override;

    bool isIntersectsWithRay( const Ray& ray ) const override;

    Vec3d getSamplePoint() const override;
private:
    Vec3d origin;
};


#endif //COLLECTION_POINTLIGHT_H
