#pragma once
#include "Vector3f.h"
#include "Ray.h"
class Triangle {
public:
    Triangle();
    Triangle( const Vector3f& v1, const Vector3f& v2, const Vector3f& v3 );
    [[nodiscard]] bool isContainPoint( const Vector3f& p ) const;
    [[nodiscard]] float intersectsWithRay( const Ray& ray ) const;
    [[nodiscard]] Vector3f getNormal() const;
private:
    Vector3f v1;
    Vector3f v2;
    Vector3f v3;
    Vector3f N;
};
