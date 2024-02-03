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
    void rotate( const Vector3f& axis, float angle ) override;
    void move( const Vector3f& p ) override;
    void moveTo( const Vector3f& point ) override;
    void scale( float scaleValue ) override;
    void scale( const Vector3f& scaleVec ) override;
    void scaleTo( float scaleValue ) override;
    void scaleTo( const Vector3f& scaleVec ) override;
    [[nodiscard]] BBoxData getBBox() const override;
    [[nodiscard]] Vector3f getOrigin() const override;
    [[nodiscard]] bool isContainPoint( const Vector3f& p ) const override;
    [[nodiscard]] IntersectionData intersectsWithRay( const Ray& ray ) const override;
    [[nodiscard]] Vector3f getNormal( const Vector3f& p ) const override;
    std::vector<Triangle> triangles;
};

