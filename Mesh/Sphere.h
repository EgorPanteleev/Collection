//
// Created by auser on 5/12/24.
//

#ifndef COLLECTION_SPHERE_H
#define COLLECTION_SPHERE_H

#include "Vector.h"
#include "Vector3f.h"
#include "Ray.h"
#include "Material.h"
#include "BBoxData.h"
#include "IntersectionData.h"

class Material;
class Sphere {
public:
    Sphere();
    Sphere( float r, const Vector3f& pos );
    Sphere( float r, const Vector3f& pos, const Material& m );
    void rotate( const Vector3f& axis, float angle );
    void move( const Vector3f& p );
    void moveTo( const Vector3f& point );
    void scale( float scaleValue );
    void scale( const Vector3f& scaleVec );
    void scaleTo( float scaleValue );
    void scaleTo( const Vector3f& scaleVec );
    [[nodiscard]] Vector3f getSamplePoint();
    [[nodiscard]] BBoxData getBBox() const;
    [[nodiscard]] Material getMaterial() const;
    [[nodiscard]] Vector3f getOrigin() const;
    [[nodiscard]] bool isContainPoint( const Vector3f& p ) const;
    [[nodiscard]] Vector3f getNormal( const Vector3f& p ) const;
    [[nodiscard]] float intersectsWithRay( const Ray& ray ) const;

public:
    Material material;
    float radius;
    Vector3f origin;
};


#endif //COLLECTION_SPHERE_H
