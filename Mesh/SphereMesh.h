//
// Created by auser on 5/12/24.
//

#ifndef COLLECTION_SPHEREMESH_H
#define COLLECTION_SPHEREMESH_H

#include "Vector.h"
#include "BaseMesh.h"
class SphereMesh: public BaseMesh {
public:
    SphereMesh();
    SphereMesh( float r, const Vector3f& pos );
    SphereMesh( float r, const Vector3f& pos, const Material& m );
    void rotate( const Vector3f& axis, float angle ) override;
    void move( const Vector3f& p ) override;
    void moveTo( const Vector3f& point ) override;
    void scale( float scaleValue ) override;
    void scale( const Vector3f& scaleVec ) override;
    void scaleTo( float scaleValue ) override;
    void scaleTo( const Vector3f& scaleVec ) override;
    [[nodiscard]] BBoxData getBBox() const override;
    [[nodiscard]] Vector3f getOrigin() const override;
    [[nodiscard]] bool isContainPoint( const Vector3f& p ) const override;
    [[nodiscard]] Vector3f getNormal( const Vector3f& p ) const override;
    [[nodiscard]] IntersectionData intersectsWithRay( const Ray& ray ) const override;

public:
    float radius;
    Vector3f origin;
};


#endif //COLLECTION_SPHEREMESH_H
