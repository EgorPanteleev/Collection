#pragma once
#include "Vector3f.h"
#include "Ray.h"
#include "BaseMesh.h"
#include "Material.h"
class BBoxData;
class BaseMesh;
class Triangle {
public:
    __host__ __device__ Triangle();
    __host__ __device__ Triangle( const Vector3f& v1, const Vector3f& v2, const Vector3f& v3 );
    __host__ __device__ void rotate( const Vector3f& axis, float angle );
    __host__ __device__ void move( const Vector3f& p );
    __host__ __device__ void moveTo( const Vector3f& point );
    __host__ __device__ void scale( float scaleValue );
    __host__ __device__ void scale( const Vector3f& scaleVec );
    __host__ __device__ void scaleTo( float scaleValue );
    __host__ __device__ void scaleTo( const Vector3f& scaleVec );
    [[nodiscard]] __host__ __device__ BBoxData getBBox() const;
    [[nodiscard]] __host__ __device__ Vector3f getOrigin() const;
    [[nodiscard]] __host__ __device__ bool isContainPoint( const Vector3f& p ) const;
    [[nodiscard]] __host__ __device__ float intersectsWithRay( const Ray& ray ) const;
    [[nodiscard]] __host__ __device__ Vector3f getNormal() const;
    Vector3f v1, v2, v3;
    Material material;
    BaseMesh* owner;
private:
    Vector3f edge1, edge2;
    Vector3f origin;
    Vector3f N;
};
