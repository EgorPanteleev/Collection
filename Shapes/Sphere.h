#ifndef COLLECTION_SPHERE_H
#define COLLECTION_SPHERE_H

#include "Vector.h"
#include "Shape.h"
class Sphere: public Shape {
public:
    Sphere();
    Sphere( float r, const Vector3f& pos, const RGB& _color );
    [[nodiscard]] bool isContainPoint( Vector3f p ) const override;
    [[nodiscard]] Vector3f getNormal( Vector3f p ) const override;
    [[nodiscard]] float intersectsWithRay( const Ray& ray ) const override;
public:
    float radius;
    Vector3f origin;
};


#endif //COLLECTION_SPHERE_H
