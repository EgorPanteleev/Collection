#include "BVH.h"

BVH::BVH( std::vector <Triangle> _triangles): triangles( _triangles ) {
}

BVH::BVH(): triangles() {
}

void BVH::BuildBVH() {
    for( int i = 0; i < triangles.size(); i++ ) bvhNode.push_back({});
    // populate triangle index array
    for (int i = 0; i < triangles.size(); i++) indexes.push_back(i);
    // assign all triangle to root node
    BVHNode& root = bvhNode[rootNodeIdx];
    root.leftFirst = 0, root.trianglesCount = triangles.size();
    Vector3f centroidMin, centroidMax;
    UpdateNodeBounds( 0, centroidMin, centroidMax );
    // subdivide recursively
    Subdivide( 0, 0, nodesUsed, centroidMin, centroidMax );
}

void BVH::UpdateNodeBounds( uint nodeIdx, Vector3f& centroidMin, Vector3f& centroidMax )
{
    BVHNode& node = bvhNode[nodeIdx];

    node.aabbMin = Vector3f( 1e30f, 1e30f, 1e30f );
    node.aabbMax = Vector3f( -1e30f, -1e30f, -1e30f );
    centroidMin = Vector3f( 1e30f, 1e30f, 1e30f );
    centroidMax = Vector3f( -1e30f, -1e30f, -1e30f );
    for (uint first = node.leftFirst, i = 0; i < node.trianglesCount; i++)
    {
        uint leafTriIdx = indexes[first + i];
        Triangle& leafTri = triangles[leafTriIdx];
        node.aabbMin = min( node.aabbMin, leafTri.v1 );
        node.aabbMin = min( node.aabbMin, leafTri.v2 );
        node.aabbMin = min( node.aabbMin, leafTri.v3 );
        node.aabbMax = max( node.aabbMax, leafTri.v1 );
        node.aabbMax = max( node.aabbMax, leafTri.v2 );
        node.aabbMax = max( node.aabbMax, leafTri.v3 );
        centroidMin = min( centroidMin, leafTri.getOrigin() );
        centroidMax = max( centroidMax, leafTri.getOrigin() );
    }
}

struct aabb
{
    Vector3f bmin = { 1e30f, 1e30f , 1e30f }, bmax = { -1e30f, -1e30f , -1e30f };
    void grow( Vector3f p ) { bmin = min( bmin, p ); bmax = max( bmax, p ); }
    void grow( aabb& b ) { if (b.bmin.x != 1e30f) { grow( b.bmin ); grow( b.bmax ); } }
    float area()
    {
        Vector3f e = bmax - bmin; // box extent
        return e.x * e.y + e.y * e.z + e.z * e.x;
    }
};

float BVH::EvaluateSAH( BVHNode& node, int axis, float pos ) {
    // determine triangle counts and bounds for this split candidate
    aabb leftBox, rightBox;
    int leftCount = 0, rightCount = 0;
    for (uint i = 0; i < node.trianglesCount; i++)
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

void BVH::Subdivide( uint nodeIdx, uint depth, uint& nodePtr, Vector3f& centroidMin, Vector3f& centroidMax )
{
    BVHNode& node = bvhNode[nodeIdx];
    if (node.trianglesCount <= 2) return;
    // determine split axis using SAH
    int axis = 1, splitPos = 1;
    float splitCost = FindBestSplitPlane( node, axis, splitPos, centroidMin, centroidMax );
    float nosplitCost = node.CalculateNodeCost();
    if (splitCost >= nosplitCost) return;
    // terminate recursion
//    if (subdivToOnePrim)
//    {
//        if (node.triCount == 1) return;
//    }
//    else
//    {
//        float nosplitCost = node.CalculateNodeCost();
//        if (splitCost >= nosplitCost) return;
//    }
    // in-place partition
    int i = node.leftFirst;
    int j = i + node.trianglesCount - 1;
    float scale = BINS / (centroidMax[axis] - centroidMin[axis]);
    while (i <= j)
    {
        // use the exact calculation we used for binning to prevent rare inaccuracies
        int binIdx = std::min( BINS - 1, (int)((triangles[indexes[i]].getOrigin()[axis] - centroidMin[axis]) * scale) );
        if (binIdx < splitPos) i++; else std::swap( indexes[i], indexes[j--] );
    }
    // abort split if one of the sides is empty
    int leftCount = i - node.leftFirst;
    if (leftCount == 0 || leftCount == node.trianglesCount) return; // never happens for dragon mesh, nice
    // create child nodes
    int leftChildIdx = nodePtr++;
    int rightChildIdx = nodePtr++;
    bvhNode[leftChildIdx].leftFirst = node.leftFirst;
    bvhNode[leftChildIdx].trianglesCount = leftCount;
    bvhNode[rightChildIdx].leftFirst = i;
    bvhNode[rightChildIdx].trianglesCount = node.trianglesCount - leftCount;
    node.leftFirst = leftChildIdx;
    node.trianglesCount = 0;
    // recurse
    UpdateNodeBounds( leftChildIdx, centroidMin, centroidMax );
    Subdivide( leftChildIdx, depth + 1, nodePtr, centroidMin, centroidMax );
    UpdateNodeBounds( rightChildIdx, centroidMin, centroidMax );
    Subdivide( rightChildIdx, depth + 1, nodePtr, centroidMin, centroidMax );
}

float BVH::FindBestSplitPlane( BVHNode& node, int& axis, int& splitPos, Vector3f& centroidMin, Vector3f& centroidMax )
{
    float bestCost = 1e30f;
    for (int a = 0; a < 3; a++)
    {
        float boundsMin = centroidMin[a], boundsMax = centroidMax[a];
        if (boundsMin == boundsMax) continue;
        // populate the bins
        float scale = BINS / (boundsMax - boundsMin);
        float leftCountArea[BINS - 1], rightCountArea[BINS - 1];
        int leftSum = 0, rightSum = 0;

        struct Bin { aabb bounds; int trianglesCount = 0; } bin[BINS];
        for (uint i = 0; i < node.trianglesCount; i++)
        {
            Triangle& triangle = triangles[indexes[node.leftFirst + i]];
            int binIdx =std:: min( BINS - 1, (int)((triangle.getOrigin()[a] - boundsMin) * scale) );
            bin[binIdx].trianglesCount++;
            bin[binIdx].bounds.grow( triangle.v1 );
            bin[binIdx].bounds.grow( triangle.v2 );
            bin[binIdx].bounds.grow( triangle.v3 );
        }
        // gather data for the 7 planes between the 8 bins
        aabb leftBox, rightBox;
        for (int i = 0; i < BINS - 1; i++)
        {
            leftSum += bin[i].trianglesCount;
            leftBox.grow( bin[i].bounds );
            leftCountArea[i] = leftSum * leftBox.area();
            rightSum += bin[BINS - 1 - i].trianglesCount;
            rightBox.grow( bin[BINS - 1 - i].bounds );
            rightCountArea[BINS - 2 - i] = rightSum * rightBox.area();
        }
        // calculate SAH cost for the 7 planes
        scale = (boundsMax - boundsMin) / BINS;
        for (int i = 0; i < BINS - 1; i++)
        {
            const float planeCost = leftCountArea[i] + rightCountArea[i];
            if (planeCost < bestCost)
                axis = a, splitPos = i + 1, bestCost = planeCost;
        }
    }
    return bestCost;
}


bool BVH::IntersectAABB( const Ray& ray, const Vector3f bmin, const Vector3f bmax )
{
    float tx1 = (bmin.x - ray.origin.x) / ray.direction.x, tx2 = (bmax.x - ray.origin.x) / ray.direction.x;
    float tmin = std::min( tx1, tx2 ), tmax = std::max( tx1, tx2 );
    float ty1 = (bmin.y - ray.origin.y) / ray.direction.y, ty2 = (bmax.y - ray.origin.y) / ray.direction.y;
    tmin = std::max( tmin, std::min( ty1, ty2 ) ), tmax = std::min( tmax, std::max( ty1, ty2 ) );
    float tz1 = (bmin.z - ray.origin.z) / ray.direction.z, tz2 = (bmax.z - ray.origin.z) / ray.direction.z;
    tmin = std::max( tmin, std::min( tz1, tz2 ) ), tmax = std::min( tmax, std::max( tz1, tz2 ) );
    return tmax >= tmin && tmin < 1e30f && tmax > 0;
}

IntersectionData BVH::IntersectBVH( Ray& ray, const uint nodeIdx )
{
    static float MAX = std::numeric_limits<float>::max();
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
}
