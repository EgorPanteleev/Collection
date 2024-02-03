#ifndef COLLECTION_SHAPE_H
#define COLLECTION_SHAPE_H
#include <iostream>
#include "Ray.h"

struct IntersectionData {
    IntersectionData(): t( std::numeric_limits<float>::max() ), N() {};
    IntersectionData( float t, const Vector3f& N ): t( t ), N( N ) {};
    float t;
    Vector3f N;
};

struct BBoxData {
    BBoxData( const Vector3f& pMin, const Vector3f& pMax ): pMin( pMin ), pMax( pMax ) {}
    Vector3f pMin;
    Vector3f pMax;
};

class Shape {
public:
    virtual void rotate( const Vector3f& axis, float angle ) = 0;
    virtual void move( const Vector3f& p ) = 0;
    virtual void moveTo( const Vector3f& point ) = 0;
    virtual void scale( float scaleValue ) = 0;
    virtual void scale( const Vector3f& scaleVec ) = 0;
    virtual void scaleTo( float scaleValue ) = 0;
    virtual void scaleTo( const Vector3f& scaleVec ) = 0;
    [[nodiscard]] virtual BBoxData getBBox() const = 0;
    [[nodiscard]] virtual Vector3f getOrigin() const = 0;
    [[nodiscard]] virtual bool isContainPoint( const Vector3f& p ) const = 0;
    [[nodiscard]] virtual IntersectionData intersectsWithRay( const Ray& ray ) const = 0;
    [[nodiscard]] virtual Vector3f getNormal( const Vector3f& p ) const = 0;
private:
};

#endif //COLLECTION_SHAPE_H
