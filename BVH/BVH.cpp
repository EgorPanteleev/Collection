#include "BVH.h"

BVH::BVH( const Vector <Primitive*>& _primitives ) {
//    for ( auto prim: _primitives ){
//        std::cout<< "Origin - "<<prim->getOrigin() << std::endl;
//        std::cout<< "BBox - "<<prim->getBBox().pMin << " " << prim->getBBox().pMax << std::endl;
//        Material mat = prim->getMaterial();
//        std::cout<< "Metalness - "<< mat.getMetalness() << std::endl;
//        std::cout<< "Roughness - "<< mat.getRoughness() << std::endl;
//        std::cout<< "Color - "<< mat.getColor().r << " " << mat.getColor().g << " " << mat.getColor().b << std::endl;
//        std::cout<< "Intensity - "<< mat.getIntensity() << std::endl;
//        std::cout<< "Diffuse - "<< mat.getDiffuse() << std::endl;
//    }
    primitives = _primitives;
    build();
}

BVH::BVH(): primitives() {
}

void BVH::build() {
    for( int i = 0; i < primitives.size(); i++ ) bvhNode.push_back({});
    for (int i = 0; i < primitives.size(); i++) indexes.push_back(i);
    BVHNode& root = bvhNode[rootNodeIdx];
    root.leftFirst = 0, root.trianglesCount = primitives.size();
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
        Primitive* prim = primitives[indexes[first + i]];
        BBox bbox = prim->getBBox();
        node.aabbMin = min( node.aabbMin, bbox.pMin );
        node.aabbMax = max( node.aabbMax, bbox.pMax );
        centroidMin = min( centroidMin, prim->getOrigin() );
        centroidMax = max( centroidMax, prim->getOrigin() );
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
        Vector3f origin = primitives[ indexes[i] ]->getOrigin();
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
            Primitive* prim = primitives[indexes[node.leftFirst + i]];
            int binIdx =std:: min( BINS - 1, (int)((prim->getOrigin()[a] - boundsMin) * scale) );
            bin[binIdx].trianglesCount++;
            BBox bbox = prim->getBBox();
            bin[binIdx].bounds.merge( bbox.pMin );
            bin[binIdx].bounds.merge( bbox.pMax );
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
    Vector3f t1 = ( bmin - ray.origin ) * ray.invDirection;
    Vector3f t2 = ( bmax - ray.origin ) * ray.invDirection;
    float tmin = std::min( t1.x, t2.x );
    tmin = std::max( tmin, std::min( t1.y, t2.y ) );
    tmin = std::max( tmin, std::min( t1.z, t2.z ) );
    float tmax = std::max( t1.x, t2.x );
    tmax = std::min( tmax, std::max( t1.y, t2.y ) );
    tmax = std::min( tmax, std::max( t1.z, t2.z ) );
    return tmax >= tmin && tmin < 1e30f && tmax > 0;
}

IntersectionData BVH::intersectBVH( Ray& ray, const uint nodeIdx ) {
    BVHNode& node = bvhNode[nodeIdx];
    if (!intersectBBox( ray, node.aabbMin, node.aabbMax )) return { __FLT_MAX__, nullptr };
    if (node.isLeaf()) {
        IntersectionData iData;
        for (uint i = 0; i < node.trianglesCount; i++ ) {
            Primitive* prim = primitives[indexes[node.leftFirst + i]];
            float t = prim->intersectsWithRay( ray );
            if ( t >= iData.t ) continue;
            iData.t = t;
            iData.primitive = prim;
        }
        return iData;
    }
    IntersectionData iData1 = intersectBVH( ray, node.leftFirst );
    IntersectionData iData2 = intersectBVH( ray, node.leftFirst + 1 );
    if ( iData1.t < iData2.t ) return iData1;
    else return iData2;
}
