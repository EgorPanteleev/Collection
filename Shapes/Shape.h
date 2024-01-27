#ifndef COLLECTION_SHAPE_H
#define COLLECTION_SHAPE_H
#include <iostream>
#include "Ray.h"
#include "Color.h"
class Shape {
public:
    [[nodiscard]] virtual bool isContainPoint( const Vector3f& p ) const = 0;
    [[nodiscard]] virtual float intersectsWithRay( const Ray& ray ) const = 0;
    [[nodiscard]] virtual Vector3f getNormal( const Vector3f& p ) const = 0;
    [[nodiscard]] RGB getColor() const;
    void setColor( const RGB& c );
private:
    RGB color;
};

#endif //COLLECTION_SHAPE_H
