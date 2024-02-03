#pragma once
#include "Vector3f.h"
#include "Ray.h"
#include "Shape.h"
class Triangle {
public:
    Triangle();
    Triangle( const Vector3f& v1, const Vector3f& v2, const Vector3f& v3 );
    void rotate( const Vector3f& axis, float angle );
    void move( const Vector3f& p );
    void moveTo( const Vector3f& point );
    void scale( float scaleValue );
    void scale( const Vector3f& scaleVec );
    void scaleTo( float scaleValue );
    void scaleTo( const Vector3f& scaleVec );
    [[nodiscard]] BBoxData getBBox() const;
    [[nodiscard]] Vector3f getOrigin() const;
    [[nodiscard]] bool isContainPoint( const Vector3f& p ) const;
    [[nodiscard]] float intersectsWithRay( const Ray& ray ) const;
    [[nodiscard]] Vector3f getNormal() const;
private:
    Vector3f v1;
    Vector3f v2;
    Vector3f v3;
    Vector3f N;
};
