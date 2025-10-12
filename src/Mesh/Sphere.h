//
// Created by auser on 5/12/24.
//

#ifndef COLLECTION_SPHERE_H
#define COLLECTION_SPHERE_H

#include "Vector.h"
#include "Vec3.h"
#include "Ray.h"
#include "Material.h"
#include "BBox.h"
#include "Primitive.h"

class Sphere: public Primitive {
public:
    Sphere();
    Sphere( double r, const Vec3d& pos );
    Sphere( double r, const Vec3d& pos, const Material& m );
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
    [[nodiscard]] int getIndex( const Vec3d& P, const ImageData& imageData ) const override;
    [[nodiscard]] Vec3d getNormal( const Vec3d& p ) const override;

public:
    double radius;
};


#endif //COLLECTION_SPHERE_H
