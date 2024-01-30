#pragma once

#include <vector>
#include "Shape.h"
#include "Vector.h"
#include "Triangle.h"

class Cube: public Shape {
public:
    Cube();
    Cube( const Vector3f& _p1, const Vector3f& _p2);
    [[nodiscard]] bool isContainPoint( const Vector3f& p ) const override;
    [[nodiscard]] IntersectionData intersectsWithRay( const Ray& ray ) const override;
    [[nodiscard]] Vector3f getNormal( const Vector3f& p ) const override;
private:
    void fillTriangles();
    Vector3f p1;
    Vector3f p2;
    std::vector<Triangle> triangles;
};


