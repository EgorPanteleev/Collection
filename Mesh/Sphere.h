//
// Created by auser on 5/12/24.
//

#ifndef COLLECTION_SPHERE_H
#define COLLECTION_SPHERE_H

#include "Vector.h"
#include "Vector3f.h"
#include "Ray.h"
#include "Material.h"
#include "BBox.h"
#include "Primitive.h"

class Sphere: public Primitive {
public:
    Sphere();
    Sphere( float r, const Vector3f& pos );
    Sphere( float r, const Vector3f& pos, const Material& m );
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
    [[nodiscard]] int getIndex( const Vector3f& P, const ImageData& imageData ) const override;
    [[nodiscard]] Vector3f getNormal( const Vector3f& p ) const override;

public:
    float radius;
};


#endif //COLLECTION_SPHERE_H
