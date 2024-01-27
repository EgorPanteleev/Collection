
#ifndef COLLECTION_CUBE_H
#define COLLECTION_CUBE_H
#include "Shape.h"
#include "Vector.h"
class Cube: public Shape {
public:
    Cube();
    Cube( const Vector3f& _p1, const Vector3f& _p2);
    [[nodiscard]] bool isContainPoint( const Vector3f& p ) const override;
    Cube( float x1, float y1, float z1, float x2, float y2, float z2);
    [[nodiscard]] float intersectsWithRay( const Ray& ray ) const override;
    [[nodiscard]] Vector3f getNormal( const Vector3f& p ) const override;
private:
    Vector3f p1;
    Vector3f p2;
};


#endif //COLLECTION_CUBE_H
