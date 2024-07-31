//
// Created by auser on 5/12/24.
//

#ifndef COLLECTION_SPHEREMESH_H
#define COLLECTION_SPHEREMESH_H

#include "Vector.h"
#include "BaseMesh.h"
class SphereMesh: public BaseMesh {
public:
    __host__ __device__ SphereMesh();
    __host__ __device__ SphereMesh( float r, const Vector3f& pos );
    __host__ __device__ SphereMesh( float r, const Vector3f& pos, const Material& m );
    __host__ __device__ void rotate( const Vector3f& axis, float angle ) override;
    __host__ __device__ void move( const Vector3f& p ) override;
    __host__ __device__ void moveTo( const Vector3f& point ) override;
    __host__ __device__ void scale( float scaleValue ) override;
    __host__ __device__ void scale( const Vector3f& scaleVec ) override;
    __host__ __device__ void scaleTo( float scaleValue ) override;
    __host__ __device__ void scaleTo( const Vector3f& scaleVec ) override;
    [[nodiscard]] __host__ __device__ BBoxData getBBox() const override;
    [[nodiscard]] __host__ __device__ Vector3f getOrigin() const override;
    [[nodiscard]] __host__ __device__ bool isContainPoint( const Vector3f& p ) const override;
    [[nodiscard]] __host__ __device__ Vector3f getNormal( const Vector3f& p ) const override;
    [[nodiscard]] __host__ __device__ IntersectionData intersectsWithRay( const Ray& ray ) const override;

public:
    float radius;
    Vector3f origin;
};


#endif //COLLECTION_SPHEREMESH_H
