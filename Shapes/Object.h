#pragma once
#include "Shape.h"
#include "Material.h"
class Object {
public:
    Object();
    Object( Shape* shape, const Material& material );
    ~Object();
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
    [[nodiscard]] IntersectionData intersectsWithRay( const Ray& ray ) const;
    [[nodiscard]] Vector3f getNormal( const Vector3f& p ) const;
    [[nodiscard]] RGB getColor() const;
    void setColor( const RGB& c );
    [[nodiscard]] float getDiffuse() const;
    void setDiffuse( float d );
    [[nodiscard]] float getReflection() const;
    void setReflection( float r );
private:
    Shape* shape;
    Material material;
};

