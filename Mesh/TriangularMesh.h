//
// Created by auser on 5/12/24.
//

#ifndef COLLECTION_TRIANGULARMESH_H
#define COLLECTION_TRIANGULARMESH_H

#include "Vector3f.h"
#include "Ray.h"
#include "BaseMesh.h"


class BBoxData;
class TriangularMesh: public BaseMesh {
public:
//    TriangularMesh( Vector<Triangle> _triangles );
//    TriangularMesh( const std::string& path );
    void loadMesh( const std::string& path ) override;
    __host__ __device__ void rotate( const Vector3f& axis, float angle ) override;
    __host__ __device__ void move( const Vector3f& p ) override;
    __host__ __device__ void moveTo( const Vector3f& point ) override;
    __host__ __device__ void scale( float scaleValue ) override;
    __host__ __device__ void scale( const Vector3f& scaleVec ) override;
    __host__ __device__ void scaleTo( float scaleValue ) override;
    __host__ __device__ void scaleTo( const Vector3f& scaleVec ) override;

    __host__ __device__ void setMinPoint( const Vector3f& vec, int ind = -1 ) override;

    __host__ __device__ void setMaxPoint( const Vector3f& vec, int ind = -1 ) override;

    __host__ __device__ Vector<Triangle> getTriangles() override;
    [[nodiscard]] __host__ __device__ BBoxData getBBox() const override;
    [[nodiscard]] __host__ __device__ Vector3f getOrigin() const override;
    [[nodiscard]] __host__ __device__ bool isContainPoint( const Vector3f& p ) const override;
    [[nodiscard]] __host__ __device__ IntersectionData intersectsWithRay( const Ray& ray ) const override;
    [[nodiscard]] __host__ __device__ Vector3f getNormal( const Vector3f& p ) const override;
    __host__ __device__ void setTriangles( Vector<Triangle>& _triangles );
    __host__ __device__ void addTriangle( const Triangle& triangle );
protected:
    Vector<Triangle> triangles;
};

#endif //COLLECTION_TRIANGULARMESH_H