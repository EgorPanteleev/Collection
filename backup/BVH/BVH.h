//
// Created by auser on 5/11/24.
//

#ifndef COLLECTION_BVH_H
#define COLLECTION_BVH_H
#define BINS 8
#include <cmath>
#include "Vector3f.h"
#include "Triangle.h"
#include <vector>
#include "Utils.h"

struct BVHNode
{
    Vector3f aabbMin, aabbMax;
    uint leftFirst, trianglesCount;

    [[nodiscard]] bool isLeaf() const {
        return ( trianglesCount > 0 );
    }
    float CalculateNodeCost()
    {
        Vector3f e = aabbMax - aabbMin; // extent of the node
        return (e.x * e.y + e.y * e.z + e.z * e.x) * trianglesCount;
    }
};


class BVH {
public:

    BVH( std::vector <Triangle> _triangles );

    void BuildBVH();

    void UpdateNodeBounds( uint nodeIdx, Vector3f& centroidMin, Vector3f& centroidMax );

    float EvaluateSAH( BVHNode& node, int axis, float pos );

    void Subdivide( uint nodeIdx, uint depth, uint& nodePtr, Vector3f& centroidMin, Vector3f& centroidMax );

    float FindBestSplitPlane( BVHNode& node, int& axis, int& splitPos, Vector3f& centroidMin, Vector3f& centroidMax );

    bool IntersectAABB( const Ray& ray, const Vector3f bmin, const Vector3f bmax );

    IntersectionData IntersectBVH( Ray& ray, const uint nodeIdx );
private:
    std::vector <Triangle> triangles;
    std::vector <uint> indexes;
    std::vector<BVHNode> bvhNode;
    uint rootNodeIdx = 0, nodesUsed = 1;

};



#endif //COLLECTION_BVH_H
