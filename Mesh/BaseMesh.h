//
// Created by auser on 5/12/24.
//

#ifndef COLLECTION_BASEMESH_H
#define COLLECTION_BASEMESH_H
#include <iostream>
#include "Ray.h"
#include <limits>
#include "Triangle.h"
#include "Vector.h"
#include "IntersectionData.h"
#include "Material.h"
struct BBoxData {
    __host__ __device__ BBoxData( const Vector3f& pMin, const Vector3f& pMax ): pMin( pMin ), pMax( pMax ) {}
    Vector3f pMin;
    Vector3f pMax;
};

class Triangle;
class IntersectionData;
class BaseMesh {
public:
    __host__ __device__ BaseMesh();
    virtual void loadMesh( const std::string& path );
    __host__ __device__ virtual void rotate( const Vector3f& axis, float angle ) = 0;
    __host__ __device__ virtual void move( const Vector3f& p ) = 0;
    __host__ __device__ virtual void moveTo( const Vector3f& point ) = 0;
    __host__ __device__ virtual void scale( float scaleValue ) = 0;
    __host__ __device__ virtual void scale( const Vector3f& scaleVec ) = 0;
    __host__ __device__ virtual void scaleTo( float scaleValue ) = 0;
    __host__ __device__ virtual void scaleTo( const Vector3f& scaleVec ) = 0;

    __host__ __device__ virtual void setMinPoint( const Vector3f& vec, int ind = -1 ) {  };

    __host__ __device__ virtual void setMaxPoint( const Vector3f& vec, int ind = -1 ) {  };

    __host__ __device__ virtual Vector <Triangle> getTriangles();
    [[nodiscard]] __host__ __device__ virtual BBoxData getBBox() const = 0;
    [[nodiscard]] __host__ __device__ virtual Vector3f getOrigin() const = 0;
    [[nodiscard]] __host__ __device__ virtual bool isContainPoint( const Vector3f& p ) const = 0;
    [[nodiscard]] __host__ __device__ virtual IntersectionData intersectsWithRay( const Ray& ray ) const = 0;
    [[nodiscard]] __host__ __device__ virtual Vector3f getNormal( const Vector3f& p ) const = 0;
    __host__ __device__ void setMaterial( const Material& _material );
    [[nodiscard]] __host__ __device__ Material getMaterial() const;
protected:
    Material material;
};

#endif //COLLECTION_BASEMESH_H
