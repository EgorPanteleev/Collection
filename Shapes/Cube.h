
#ifndef COLLECTION_CUBE_H
#define COLLECTION_CUBE_H
#include "Shape.h"
#include "Vector.h"
class Cube: public Shape {
public:
    Cube();
    Cube( Vector3f _p1, Vector3f _p2);
    virtual bool isContainPoint( Vector3f p ) const;
    Cube( double x1, double y1, double z1, double x2, double y2, double z2);
    virtual double intersectsWithRay( const Ray& ray );
    virtual Vector3f getNormal( Vector3f p ){}
private:
    Vector3f p1;
    Vector3f p2;
};


#endif //COLLECTION_CUBE_H
