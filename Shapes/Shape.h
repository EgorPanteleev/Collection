#ifndef COLLECTION_SHAPE_H
#define COLLECTION_SHAPE_H
#include <iostream>
#include "Ray.h"
#include "Color.h"
class Shape {
public:
    virtual double intersectsWithRay( const Ray& ray ) = 0;
    virtual Point getNormal( Point p ) = 0;
    RGB getColor() const;
    void setColor( RGB c );
private:
    RGB color;
};

#endif //COLLECTION_SHAPE_H
