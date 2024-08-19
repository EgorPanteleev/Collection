#pragma once
#include "Vector3f.h"
#include "Vector2f.h"
#include "Ray.h"
#include "Mesh.h"
#include "Material.h"
class BBoxData;
class Mesh;
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
    [[nodiscard]] Vector3f getSamplePoint() const;
    [[nodiscard]] BBoxData getBBox() const;
    [[nodiscard]] Vector3f getOrigin() const;
    [[nodiscard]] bool isContainPoint( const Vector3f& p ) const;
    [[nodiscard]] float intersectsWithRay( const Ray& ray ) const;
    [[nodiscard]] Vector3f getNormal() const;
    [[nodiscard]] Vector3f getNormal( const Vector3f& P ) const;
    [[nodiscard]] RGB getColor( const Vector3f& P ) const;
    [[nodiscard]] RGB getAmbient( const Vector3f& P ) const;
    [[nodiscard]] float getRoughness( const Vector3f& P ) const;
    [[nodiscard]] Vector3f getBarycentric( const Vector3f& P ) const;
    Vector3f v1, v2, v3;
    Mesh* owner;
private:
    Vector2f v1Tex, v2Tex, v3Tex;
    Vector3f edge1, edge2;
    Vector3f origin;
    Vector3f N;
};
