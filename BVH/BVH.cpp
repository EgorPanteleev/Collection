#include "BVH.h"

BVH::BVH( const Vector <Triangle>& _triangles, const Vector <Sphere>& _spheres ) {
    triangles = _triangles;
    spheres = _spheres;
    build();
}

BVH::BVH(): triangles(), spheres() {
}

void BVH::build() {
    for( int i = 0; i < triangles.size() + spheres.size(); i++ ) bvhNode.push_back({});
    for (int i = 0; i < triangles.size() + spheres.size(); i++) indexes.push_back(i);
    BVHNode& root = bvhNode[rootNodeIdx];
    root.leftFirst = 0, root.trianglesCount = triangles.size() + spheres.size();
    Vector3f centroidMin, centroidMax;
    updateNodeBounds( 0, centroidMin, centroidMax );
    subDivide( 0, 0, nodesUsed, centroidMin, centroidMax );
}

void BVH::updateNodeBounds( uint nodeIdx, Vector3f& centroidMin, Vector3f& centroidMax ) {
    BVHNode& node = bvhNode[nodeIdx];
    node.aabbMin = Vector3f( 1e30f, 1e30f, 1e30f );
    node.aabbMax = Vector3f( -1e30f, -1e30f, -1e30f );
    centroidMin = Vector3f( 1e30f, 1e30f, 1e30f );
    centroidMax = Vector3f( -1e30f, -1e30f, -1e30f );
    for (uint first = node.leftFirst, i = 0; i < node.trianglesCount; i++) {
        uint leafTriIdx = indexes[first + i];
        if ( leafTriIdx >= triangles.size() ) {
            leafTriIdx -= triangles.size();
            Sphere& leafSphe = spheres[leafTriIdx];
            BBox bbox = leafSphe.getBBox();
            node.aabbMin = min( node.aabbMin, bbox.pMin );
            node.aabbMax = max( node.aabbMax, bbox.pMax );
        } else {
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
}

void BVH::subDivide( uint nodeIdx, uint depth, uint& nodePtr, Vector3f& centroidMin, Vector3f& centroidMax ) {
    BVHNode& node = bvhNode[nodeIdx];
    if (node.trianglesCount <= 2) return;
    int axis = 1, splitPos = 1;
    float splitCost = findBestSplitPlane( node, axis, splitPos, centroidMin, centroidMax );
    float nosplitCost = node.calculateNodeCost();
    if (splitCost >= nosplitCost) return;
    int i = node.leftFirst;
    int j = i + node.trianglesCount - 1;
    float scale = BINS / (centroidMax[axis] - centroidMin[axis]);
    while (i <= j) {
        Vector3f origin;
        if ( indexes[i] >= triangles.size() ) origin = spheres[ indexes[i] - triangles.size() ].getOrigin();
        else origin = triangles[indexes[i]].getOrigin();
        int binIdx = std::min( BINS - 1, (int)((origin[axis] - centroidMin[axis]) * scale) );
        if (binIdx < splitPos) i++; else std::swap( indexes[i], indexes[j--] );
    }
    int leftCount = i - node.leftFirst;
    if (leftCount == 0 || leftCount == node.trianglesCount) return; // never happens for dragon mesh, nice
    int leftChildIdx = nodePtr++;
    int rightChildIdx = nodePtr++;
    bvhNode[leftChildIdx].leftFirst = node.leftFirst;
    bvhNode[leftChildIdx].trianglesCount = leftCount;
    bvhNode[rightChildIdx].leftFirst = i;
    bvhNode[rightChildIdx].trianglesCount = node.trianglesCount - leftCount;
    node.leftFirst = leftChildIdx;
    node.trianglesCount = 0;
    updateNodeBounds( leftChildIdx, centroidMin, centroidMax );
    subDivide( leftChildIdx, depth + 1, nodePtr, centroidMin, centroidMax );
    updateNodeBounds( rightChildIdx, centroidMin, centroidMax );
    subDivide( rightChildIdx, depth + 1, nodePtr, centroidMin, centroidMax );
}

float BVH::findBestSplitPlane( BVHNode& node, int& axis, int& splitPos, Vector3f& centroidMin, Vector3f& centroidMax ) {
    float bestCost = 1e30f;
    for (int a = 0; a < 3; a++) {
        float boundsMin = centroidMin[a], boundsMax = centroidMax[a];
        if (boundsMin == boundsMax) continue;
        float scale = BINS / (boundsMax - boundsMin);
        float leftCountArea[BINS - 1], rightCountArea[BINS - 1];
        int leftSum = 0, rightSum = 0;
        struct Bin { BBox bounds; int trianglesCount = 0; } bin[BINS];
        for (uint i = 0; i < node.trianglesCount; i++) {
            int leafTriIdx = indexes[node.leftFirst + i];
            if ( leafTriIdx >= triangles.size() ) {
                leafTriIdx -= triangles.size();
                Sphere& sphere = spheres[leafTriIdx];
                int binIdx =std:: min( BINS - 1, (int)((sphere.getOrigin()[a] - boundsMin) * scale) );
                bin[binIdx].trianglesCount++;
                BBox bbox = sphere.getBBox();
                bin[binIdx].bounds.merge( bbox.pMin );
                bin[binIdx].bounds.merge( bbox.pMax );
            } else {
                Triangle& triangle = triangles[leafTriIdx];
                int binIdx =std:: min( BINS - 1, (int)((triangle.getOrigin()[a] - boundsMin) * scale) );
                bin[binIdx].trianglesCount++;
                bin[binIdx].bounds.merge( triangle.v1 );
                bin[binIdx].bounds.merge( triangle.v2 );
                bin[binIdx].bounds.merge( triangle.v3 );
            }
        }
        BBox leftBox, rightBox;
        for (int i = 0; i < BINS - 1; i++) {
            leftSum += bin[i].trianglesCount;
            leftBox.merge( bin[i].bounds );
            leftCountArea[i] = leftSum * leftBox.area();
            rightSum += bin[BINS - 1 - i].trianglesCount;
            rightBox.merge( bin[BINS - 1 - i].bounds );
            rightCountArea[BINS - 2 - i] = rightSum * rightBox.area();
        }
        for (int i = 0; i < BINS - 1; i++) {
            const float planeCost = leftCountArea[i] + rightCountArea[i];
            if (planeCost < bestCost)
                axis = a, splitPos = i + 1, bestCost = planeCost;
        }
    }
    return bestCost;
}


bool BVH::intersectBBox( const Ray& ray, const Vector3f& bmin, const Vector3f& bmax ) {
    float tx1 = (bmin.x - ray.origin.x) / ray.direction.x, tx2 = (bmax.x - ray.origin.x) / ray.direction.x;
    float tmin = std::min( tx1, tx2 ), tmax = std::max( tx1, tx2 );
    float ty1 = (bmin.y - ray.origin.y) / ray.direction.y, ty2 = (bmax.y - ray.origin.y) / ray.direction.y;
    tmin = std::max( tmin, std::min( ty1, ty2 ) ), tmax = std::min( tmax, std::max( ty1, ty2 ) );
    float tz1 = (bmin.z - ray.origin.z) / ray.direction.z, tz2 = (bmax.z - ray.origin.z) / ray.direction.z;
    tmin = std::max( tmin, std::min( tz1, tz2 ) ), tmax = std::min( tmax, std::max( tz1, tz2 ) );
    return tmax >= tmin && tmin < 1e30f && tmax > 0;
}

IntersectionData BVH::intersectBVH( Ray& ray, const uint nodeIdx ) {
    BVHNode& node = bvhNode[nodeIdx];
    if (!intersectBBox( ray, node.aabbMin, node.aabbMax )) return { __FLT_MAX__, {} , nullptr, nullptr };
    if (node.isLeaf()) {
        IntersectionData iData;
        for (uint i = 0; i < node.trianglesCount; i++ ) {
            int leafTriIdx = indexes[node.leftFirst + i];
            size_t size = triangles.size();
            if ( leafTriIdx >= size ) {
                leafTriIdx -= size;
                Sphere& sphere = spheres[leafTriIdx];
                float t = sphere.intersectsWithRay( ray );
                if ( t < 0.05 || t >= iData.t ) continue;
                iData.t = t;
                iData.sphere = &sphere;
                iData.triangle = nullptr;
            } else {
                Triangle& triangle = triangles[leafTriIdx];
                float t = triangle.intersectsWithRay( ray );
                if ( t >= iData.t ) continue;
                iData.t = t;
                iData.triangle = &triangle;
                iData.sphere = nullptr;
            }
        }
        return iData;
    }
    IntersectionData iData1 = intersectBVH( ray, node.leftFirst );
    IntersectionData iData2 = intersectBVH( ray, node.leftFirst + 1 );
    if ( iData1.t < iData2.t ) return iData1;
    else return iData2;
}
