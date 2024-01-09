
#ifndef COLLECTION_CUBE_H
#define COLLECTION_CUBE_H
#include "Shape.h"
#include "Point.h"
class Cube: public Shape {
public:
    Cube();
    Cube( Point _p1, Point _p2);
    Cube( double x1, double y1, double z1, double x2, double y2, double z2);
    virtual double intersectsWithRay( const Ray& ray );
private:
    Point p1;
    Point p2;
};


#endif //COLLECTION_CUBE_H
