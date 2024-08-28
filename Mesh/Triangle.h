#pragma once
#include "Vector3f.h"
#include "Vector2f.h"
#include "Ray.h"
#include "Material.h"
#include "BBox.h"
#include "Primitive.h"

class Triangle: public Primitive {
public:
    Triangle();
    Triangle( const Vector3f& v1, const Vector3f& v2, const Vector3f& v3 );
    void rotate( const Vector3f& axis, float angle ) override;
    void move( const Vector3f& p ) override;
    void moveTo( const Vector3f& point ) override;
    void scale( float scaleValue ) override;
    void scale( const Vector3f& scaleVec ) override;
    void scaleTo( float scaleValue ) override;
    void scaleTo( const Vector3f& scaleVec ) override;
    [[nodiscard]] Vector3f getSamplePoint() const override;
    [[nodiscard]] bool isContainPoint( const Vector3f& p ) const override;
    [[nodiscard]] float intersectsWithRay( const Ray& ray ) const override;
    [[nodiscard]] Vector3f getNormal( const Vector3f& P ) const override;
    Vector3f v1, v2, v3;
private:
    [[nodiscard]] int getIndex( const Vector3f& P, const ImageData& imageData ) const override;
    Vector2f v1Tex, v2Tex, v3Tex;
    Vector3f edge1, edge2;
    Vector3f N;
};
