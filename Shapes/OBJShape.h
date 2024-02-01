#pragma once
#include <iostream>
#include "Shape.h"
#include "Vector.h"
#include "Ray.h"
#include <vector>
#include "Triangle.h"
class OBJShape: public Shape {
public:
    OBJShape( const std::string& path );
    [[nodiscard]] bool isContainPoint( const Vector3f& p ) const override;
    [[nodiscard]] IntersectionData intersectsWithRay( const Ray& ray ) const override;
    [[nodiscard]] Vector3f getNormal( const Vector3f& p ) const override;
    std::vector<Triangle> triangles;
};

