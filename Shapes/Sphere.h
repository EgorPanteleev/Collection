#ifndef COLLECTION_SPHERE_H
#define COLLECTION_SPHERE_H

#include "Vector.h"
#include "Shape.h"
class Sphere: public Shape {
public:
    Sphere();
    Sphere( double r, Vector3f pos, RGB _color );
    virtual Vector3f getNormal( Vector3f p );
    virtual double intersectsWithRay( const Ray& ray );
public:
    double radius;
    Vector3f origin;
};


#endif //COLLECTION_SPHERE_H
