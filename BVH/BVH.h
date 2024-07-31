//
// Created by auser on 5/11/24.
//

#ifndef COLLECTION_BVH_H
#define COLLECTION_BVH_H
#define BINS 8
#include <cmath>
#include "Vector3f.h"
#include "Triangle.h"
#include "Vector.h"
#include "Utils.h"

struct BVHNode
{
    Vector3f aabbMin, aabbMax;
    uint leftFirst, trianglesCount;

    [[nodiscard]] __host__ __device__ bool isLeaf() const {
        return ( trianglesCount > 0 );
    }
    __host__ __device__ float CalculateNodeCost()
    {
        Vector3f e = aabbMax - aabbMin; // extent of the node
        return (e.x * e.y + e.y * e.z + e.z * e.x) * trianglesCount;
    }
};


class BVH {
public:

    __host__ __device__ BVH( Vector <Triangle> _triangles );

    __host__ __device__ BVH();

    __host__ __device__ void BuildBVH();

    __host__ __device__ void UpdateNodeBounds( uint nodeIdx, Vector3f& centroidMin, Vector3f& centroidMax );

    __host__ __device__ float EvaluateSAH( BVHNode& node, int axis, float pos );

    __host__ __device__ void Subdivide( uint nodeIdx, uint depth, uint& nodePtr, Vector3f& centroidMin, Vector3f& centroidMax );

    __host__ __device__ float FindBestSplitPlane( BVHNode& node, int& axis, int& splitPos, Vector3f& centroidMin, Vector3f& centroidMax );

    __host__ __device__ bool IntersectAABB( const Ray& ray, const Vector3f bmin, const Vector3f bmax );

    __host__ __device__ IntersectionData IntersectBVH( Ray& ray, const uint nodeIdx );
private:

    Vector <Triangle> triangles;
    Vector <uint> indexes;
    Vector<BVHNode> bvhNode;
    uint rootNodeIdx = 0, nodesUsed = 1;
};



#endif //COLLECTION_BVH_H
