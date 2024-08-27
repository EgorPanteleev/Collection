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
#include "Sphere.h"
#include "IntersectionData.h"

struct BVHNode
{
    Vector3f aabbMin, aabbMax;
    uint leftFirst, trianglesCount;

    [[nodiscard]] bool isLeaf() const {
        return ( trianglesCount > 0 );
    }
    [[nodiscard]] float calculateNodeCost() const
    {
        Vector3f e = aabbMax - aabbMin; // extent of the node
        return (e.x * e.y + e.y * e.z + e.z * e.x) * (float) trianglesCount;
    }
};


class BVH {
public:

    BVH( const Vector <Triangle>& _triangles, const Vector <Sphere>& _spheres );

    BVH();

    void build();

    void updateNodeBounds( uint nodeIdx, Vector3f& centroidMin, Vector3f& centroidMax );

    void subDivide( uint nodeIdx, uint depth, uint& nodePtr, Vector3f& centroidMin, Vector3f& centroidMax );

    float findBestSplitPlane( BVHNode& node, int& axis, int& splitPos, Vector3f& centroidMin, Vector3f& centroidMax );

    bool intersectBBox( const Ray& ray, const Vector3f& bmin, const Vector3f& bmax );

    IntersectionData intersectBVH( Ray& ray, const uint nodeIdx );
private:
    Vector <Triangle> triangles;
    Vector <Sphere> spheres;
    Vector <uint> indexes;
    Vector<BVHNode> bvhNode;
    uint rootNodeIdx = 0, nodesUsed = 1;
};



#endif //COLLECTION_BVH_H
