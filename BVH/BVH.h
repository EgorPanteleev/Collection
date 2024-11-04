//
// Created by auser on 5/11/24.
//

#ifndef COLLECTION_BVH_H
#define COLLECTION_BVH_H
#define BINS 8
#include <cmath>
#include "Vec3.h"
#include "TriangleBuffer.h"
#include "Vector.h"
#include "IntersectionData.h"

struct BVHNode
{
    Vec3d aabbMin, aabbMax;
    uint leftFirst, trianglesCount;

    [[nodiscard]] bool isLeaf() const {
        return ( trianglesCount > 0 );
    }
    [[nodiscard]] double calculateNodeCost() const {
        Vec3d e = aabbMax - aabbMin;
        return (e[0] * e[1] + e[1] * e[2] + e[2] * e[0]) * trianglesCount;
    }
};


class BVH {
public:

    BVH( const Vector <Primitive*>& _primitives );

    BVH();

    void build();

    void updateNodeBounds( uint nodeIdx, Vec3d& centroidMin, Vec3d& centroidMax );

    void subDivide( uint nodeIdx, uint depth, uint& nodePtr, Vec3d& centroidMin, Vec3d& centroidMax );

    double findBestSplitPlane( BVHNode& node, int& axis, int& splitPos, Vec3d& centroidMin, Vec3d& centroidMax );

    bool intersectBBox( const Ray& ray, const Vec3d& bmin, const Vec3d& bmax ) const;

    void intersectBVH( const Ray& ray, IntersectionData& tData, uint nodeIdx ) const;
private:
    TriangleBuffer triangleBuffer;
    Vector <uint> indexes;
    Vector<BVHNode> bvhNode;
    uint rootNodeIdx = 0, nodesUsed = 1;
};



#endif //COLLECTION_BVH_H
