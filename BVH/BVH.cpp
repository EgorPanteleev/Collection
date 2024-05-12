#include "BVH.h"

BVH::BVH( std::vector <Triangle> _triangles): triangles( _triangles ) {
}

void BVH::BuildBVH()
{
    for( int i = 0; i < triangles.size(); i++ ) bvhNode.push_back({});
    // populate triangle index array
    for (int i = 0; i < triangles.size(); i++) indexes.push_back(i);
    // assign all triangle to root node
    BVHNode& root = bvhNode[rootNodeIdx];
    root.leftFirst = 0, root.trianglesCount = triangles.size();
    UpdateNodeBounds( rootNodeIdx );
    // subdivide recursively
    Subdivide( rootNodeIdx );
}

void BVH::UpdateNodeBounds( uint nodeIdx )
{
    BVHNode& node = bvhNode[nodeIdx];
    node.aabbMin = Vector3f( 1e30f, 1e30f,1e30f );
    node.aabbMax = Vector3f( -1e30f, -1e30f , -1e30f );
    for (uint first = node.leftFirst, i = 0; i < node.trianglesCount; i++)
    {
        uint leafindexes = indexes[first + i];
        Triangle& triangle = triangles[leafindexes];
        node.aabbMin = min( node.aabbMin, triangle.v1 ),
        node.aabbMin = min( node.aabbMin, triangle.v2 ),
        node.aabbMin = min( node.aabbMin, triangle.v3 ),
        node.aabbMax = max( node.aabbMax, triangle.v1 ),
        node.aabbMax = max( node.aabbMax, triangle.v2 ),
        node.aabbMax = max( node.aabbMax, triangle.v3 );
    }
}

struct aabb
{
    Vector3f bmin = { 1e30f, 1e30f , 1e30f }, bmax = { -1e30f, -1e30f ,-1e30f  };
    void grow( Vector3f p ) { bmin = min( bmin, p ), bmax = max( bmax, p ); }
    float area()
    {
        Vector3f e = bmax - bmin; // box extent
        return e.x * e.y + e.y * e.z + e.z * e.x;
    }
};

float BVH::EvaluateSAH( BVHNode& node, int axis, float pos )
{
// determine triangle counts and bounds for this split candidate
    aabb leftBox, rightBox;
    int leftCount = 0, rightCount = 0;
    for( uint i = 0; i < node.trianglesCount; i++ )
    {
        Triangle& triangle = triangles[indexes[node.leftFirst + i]];
        if (triangle.getOrigin()[axis] < pos)
        {
            leftCount++;
            leftBox.grow( triangle.v1 );
            leftBox.grow( triangle.v2 );
            leftBox.grow( triangle.v3 );
        }
        else
        {
            rightCount++;
            rightBox.grow( triangle.v1 );
            rightBox.grow( triangle.v2 );
            rightBox.grow( triangle.v3 );
        }
    }
    float cost = leftCount * leftBox.area() + rightCount * rightBox.area();
    return cost > 0 ? cost : 1e30f;
}

void BVH::Subdivide( uint nodeIdx )
{
    // terminate recursion
    BVHNode& node = bvhNode[nodeIdx];
    if (node.trianglesCount <= 2) return;
    // determine split axis using SAH
//    int bestAxis = -1;
//    float bestPos = 0, bestCost = 1e30f;
//    for( int axis = 0; axis < 3; axis++ ) for( uint i = 0; i < node.trianglesCount; i++ )
//        {
//            triangles& triangle = triangles[indexes[node.leftFirst + i]];
//            float candidatePos = triangle.getOrigin()[axis];
//            float cost = EvaluateSAH( node, axis, candidatePos );
//            if (cost < bestCost)
//                bestPos = candidatePos, bestAxis = axis, bestCost = cost;
//        }
//    int axis = bestAxis;
//    float splitPos = bestPos;
    // determine split axis and position
    Vector3f extent = node.aabbMax - node.aabbMin;
    int axis = 0;
    if (extent.y > extent.x) axis = 1;
    if (extent.z > extent[axis]) axis = 2;
    float splitPos = node.aabbMin[axis] + extent[axis] * 0.5f;
    // in-place partition
    int i = node.leftFirst;
    int j = i + node.trianglesCount - 1;
    while (i <= j)
    {
        if (triangles[indexes[i]].getOrigin()[axis] < splitPos)
            i++;
        else
            std::swap( indexes[i], indexes[j--] );
    }
    // abort split if one of the sides is empty
    int leftCount = i - node.leftFirst;
    if (leftCount == 0 || leftCount == node.trianglesCount) return;
    // create child nodes
    int leftChildIdx = nodesUsed++;
    int rightChildIdx = nodesUsed++;
    bvhNode[leftChildIdx].leftFirst = node.leftFirst;
    bvhNode[leftChildIdx].trianglesCount = leftCount;
    bvhNode[rightChildIdx].leftFirst = i;
    bvhNode[rightChildIdx].trianglesCount = node.trianglesCount - leftCount;
    node.leftFirst = leftChildIdx;
    node.trianglesCount = 0;
    UpdateNodeBounds( leftChildIdx );
    UpdateNodeBounds( rightChildIdx );
    // recurse
    Subdivide( leftChildIdx );
    Subdivide( rightChildIdx );
}

bool BVH::IntersectAABB( const Ray& ray, const Vector3f bmin, const Vector3f bmax )
{
    float tx1 = (bmin.x - ray.origin.x) / ray.direction.x, tx2 = (bmax.x - ray.origin.x) / ray.direction.x;
    float tmin = std::min( tx1, tx2 ), tmax = std::max( tx1, tx2 );
    float ty1 = (bmin.y - ray.origin.y) / ray.direction.y, ty2 = (bmax.y - ray.origin.y) / ray.direction.y;
    tmin = std::max( tmin, std::min( ty1, ty2 ) ), tmax = std::min( tmax, std::max( ty1, ty2 ) );
    float tz1 = (bmin.z - ray.origin.z) / ray.direction.z, tz2 = (bmax.z - ray.origin.z) / ray.direction.z;
    tmin = std::max( tmin, std::min( tz1, tz2 ) ), tmax = std::min( tmax, std::max( tz1, tz2 ) );
    return tmax >= tmin && tmin < 1e30f && tmax > 0; //was ray.t
}

IntersectionData BVH::IntersectBVH( Ray& ray, const uint nodeIdx )
{
    float MAX = std::numeric_limits<float>::max();
    BVHNode& node = bvhNode[nodeIdx];
    if (!IntersectAABB( ray, node.aabbMin, node.aabbMax )) return { MAX, {} , nullptr};
    if (node.isLeaf())
    {
        IntersectionData iData;
        for (uint i = 0; i < node.trianglesCount; i++ ) {
            Triangle triangle = triangles[indexes[node.leftFirst + i]];
            float t = triangle.intersectsWithRay( ray );
            if ( t >= iData.t ) continue;
            iData.t = t;
            iData.N = triangle.getNormal();
            iData.triangle = &(triangles[indexes[node.leftFirst + i]]);
        }
        return iData;
    }
    else
    {
        IntersectionData iData1 = IntersectBVH( ray, node.leftFirst );
        IntersectionData iData2 = IntersectBVH( ray, node.leftFirst + 1 );
        if ( iData1.t < iData2.t ) return iData1;
        else return iData2;
    }
    return { MAX, {}, nullptr };
}
