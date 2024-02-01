#ifndef COLLECTION_SPHERE_H
#define COLLECTION_SPHERE_H

#include "Vector.h"
#include "Shape.h"
class Sphere: public Shape {
public:
    Sphere();
    Sphere( double r, const Vector3f& pos );
    void move( const Vector3f& p ) override;
    [[nodiscard]] bool isContainPoint( const Vector3f& p ) const override;
    [[nodiscard]] Vector3f getNormal( const Vector3f& p ) const override;
    [[nodiscard]] IntersectionData intersectsWithRay( const Ray& ray ) const override;
public:
    double radius;
    Vector3f origin;
};


#endif //COLLECTION_SPHERE_H
