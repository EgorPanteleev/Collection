#pragma once
#include "Vec3.h"
#include "Vec2.h"
#include "Ray.h"
#include "Material.h"
#include "BBox.h"
#include "Primitive.h"

class Triangle: public Primitive {
public:
    Triangle();
    Triangle( const Vec3d& v1, const Vec3d& v2, const Vec3d& v3 );
    void rotate( const Vec3d& axis, double angle ) override;
    void move( const Vec3d& p ) override;
    void moveTo( const Vec3d& point ) override;
    void scale( double scaleValue ) override;
    void scale( const Vec3d& scaleVec ) override;
    void scaleTo( double scaleValue ) override;
    void scaleTo( const Vec3d& scaleVec ) override;
    [[nodiscard]] Vec3d getSamplePoint() const override;
    [[nodiscard]] bool isContainPoint( const Vec3d& p ) const override;
    [[nodiscard]] double intersectsWithRay( const Ray& ray ) const override;
    [[nodiscard]] Vec3d getNormal( const Vec3d& P ) const override;
    [[nodiscard]] Vec3d getV1() const override { return v1; }
    [[nodiscard]] Vec3d getV2() const override { return v2; }
    [[nodiscard]] Vec3d getV3() const override { return v3; }
    Vec3d v1, v2, v3;
private:
    [[nodiscard]] int getIndex( const Vec3d& P, const ImageData& imageData ) const override;
    Vec2d v1Tex, v2Tex, v3Tex;
    Vec3d edge1, edge2;
    Vec3d N;
};
