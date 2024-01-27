#ifndef COLLECTION_SHAPE_H
#define COLLECTION_SHAPE_H
#include <iostream>
#include "Ray.h"
#include "Color.h"
class Shape {
public:
    virtual bool isContainPoint( Vector3f p ) const = 0;
    virtual double intersectsWithRay( const Ray& ray ) = 0;
    virtual Vector3f getNormal( Vector3f p ) = 0;
    RGB getColor() const;
    void setColor( RGB c );
private:
    RGB color;
};

#endif //COLLECTION_SHAPE_H
