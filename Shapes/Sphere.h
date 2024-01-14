#ifndef COLLECTION_SPHERE_H
#define COLLECTION_SPHERE_H

#include "Point.h"
#include "Shape.h"
class Sphere: public Shape {
public:
    Sphere();
    Sphere( double r, Point pos, RGB _color );
    virtual Point getNormal( Point p );
    virtual double intersectsWithRay( const Ray& ray );
public:
    double radius;
    Point origin;
};


#endif //COLLECTION_SPHERE_H
