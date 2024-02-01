#pragma once
#include "Shape.h"
#include "Material.h"
class Object {
public:
    Object();
    Object( Shape* shape, const Material& material );
    ~Object();
    void move( const Vector3f& p );
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

