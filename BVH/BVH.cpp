#include "BVH.h"
#include "Triangle.h"
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
    for ( auto primitive: _primitives ) {
        triangleBuffer.addTriangle( primitive->getMaterial(), primitive->getV1(), primitive->getV2(), primitive->getV3() );
    }
    build();
}

BVH::BVH(): triangleBuffer() {
}

void BVH::build() {
    bvhNode.resize( triangleBuffer.size() );
    for ( size_t i = 0; i < triangleBuffer.size(); ++i) indexes.push_back(i);
    BVHNode& root = bvhNode[rootNodeIdx];
    root.leftFirst = 0, root.trianglesCount = triangleBuffer.size();
    Vec3d centroidMin, centroidMax;
    updateNodeBounds( 0, centroidMin, centroidMax );
    subDivide( 0, 0, nodesUsed, centroidMin, centroidMax );
}

void BVH::updateNodeBounds( uint nodeIdx, Vec3d& centroidMin, Vec3d& centroidMax ) {
    BVHNode& node = bvhNode[nodeIdx];
    node.aabbMin = Vec3d( 1e30, 1e30, 1e30 );
    node.aabbMax = Vec3d( -1e30, -1e30, -1e30 );
    centroidMin = Vec3d( 1e30, 1e30, 1e30 );
    centroidMax = Vec3d( -1e30, -1e30, -1e30 );
    for (uint first = node.leftFirst, i = 0; i < node.trianglesCount; i++) {
        BBox bbox = triangleBuffer.getBBox( indexes[first + i] );
        node.aabbMin = min( node.aabbMin, bbox.pMin );
        node.aabbMax = max( node.aabbMax, bbox.pMax );
       // if ( primitives[ indexes[first + i] ]->getOrigin() != triangles.getOrigin( indexes[first + i] ) )
       // { std::cerr << "origin" << std::endl;}
        centroidMin = min( centroidMin, triangleBuffer.getOrigin( indexes[first + i] ) );
        centroidMax = max( centroidMax, triangleBuffer.getOrigin( indexes[first + i] ) );
    }
}

void BVH::subDivide( uint nodeIdx, uint depth, uint& nodePtr, Vec3d& centroidMin, Vec3d& centroidMax ) {
    BVHNode& node = bvhNode[nodeIdx];
    if (node.trianglesCount <= 2) return;
    int axis = 1, splitPos = 1;
    double splitCost = findBestSplitPlane( node, axis, splitPos, centroidMin, centroidMax );
    double nosplitCost = node.calculateNodeCost();
    if (splitCost >= nosplitCost) return;
    int i = node.leftFirst;
    int j = i + node.trianglesCount - 1;
    double scale = BINS / (centroidMax[axis] - centroidMin[axis]);
    while (i <= j) {
        Vec3d origin = triangleBuffer.getOrigin( indexes[i] );
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

double BVH::findBestSplitPlane( BVHNode& node, int& axis, int& splitPos, Vec3d& centroidMin, Vec3d& centroidMax ) {
    double bestCost = 1e30;
    for (int a = 0; a < 3; a++) {
        double boundsMin = centroidMin[a], boundsMax = centroidMax[a];
        if (boundsMin == boundsMax) continue;
        double scale = BINS / (boundsMax - boundsMin);
        double leftCountArea[BINS - 1], rightCountArea[BINS - 1];
        int leftSum = 0, rightSum = 0;
        struct Bin { BBox bounds; int trianglesCount = 0; } bin[BINS];
        for (uint i = 0; i < node.trianglesCount; i++) {
            int binIdx =std:: min( BINS - 1, (int)((triangleBuffer.getOrigin( indexes[node.leftFirst + i] )[a] - boundsMin) * scale) );
            bin[binIdx].trianglesCount++;
            BBox bbox = triangleBuffer.getBBox( indexes[node.leftFirst + i] );
            bin[binIdx].bounds.merge( bbox.pMin );
            bin[binIdx].bounds.merge( bbox.pMax );
        }
        BBox leftBox, rightBox;
        for (int i = 0; i < BINS - 1; i++) {
            leftSum += bin[i].trianglesCount;
            leftBox.merge( bin[i].bounds );
            leftCountArea[i] = leftSum * leftBox.getArea();
            rightSum += bin[BINS - 1 - i].trianglesCount;
            rightBox.merge( bin[BINS - 1 - i].bounds );
            rightCountArea[BINS - 2 - i] = rightSum * rightBox.getArea();
        }
        for (int i = 0; i < BINS - 1; i++) {
            const double planeCost = leftCountArea[i] + rightCountArea[i];
            if (planeCost < bestCost)
                axis = a, splitPos = i + 1, bestCost = planeCost;
        }
    }
    return bestCost;
}


bool BVH::intersectBBox( const Ray& ray, const Vec3d& bmin, const Vec3d& bmax ) const {
    Vec3d t1 = ( bmin - ray.origin ) * ray.invDirection;
    Vec3d t2 = ( bmax - ray.origin ) * ray.invDirection;
    double tmin = std::min( t1[0], t2[0] );
    tmin = std::max( tmin, std::min( t1[1], t2[1] ) );
    tmin = std::max( tmin, std::min( t1[2], t2[2] ) );
    double tmax = std::max( t1[0], t2[0] );
    tmax = std::min( tmax, std::max( t1[1], t2[1] ) );
    tmax = std::min( tmax, std::max( t1[2], t2[2] ) );
    return tmax >= tmin && tmin < 1e30 && tmax > 0;
}

void BVH::intersectBVH( const Ray& ray, IntersectionData& tData, uint nodeIdx ) const  {
    const BVHNode& node = bvhNode[nodeIdx];
    if (!intersectBBox( ray, node.aabbMin, node.aabbMax )) {
        tData = {};
        return;
    }
    if (node.isLeaf()) {
        for (uint i = 0; i < node.trianglesCount; i++ ) {
            double t = triangleBuffer.intersectsWithRay( ray, indexes[node.leftFirst + i] );
            if ( t >= tData.t ) continue;
            tData.t = t;
            Vec4i ind = triangleBuffer.indices[indexes[node.leftFirst + i] ];
            tData.material = triangleBuffer.materials[ ind[3] ];
            tData.N = triangleBuffer.getNormal( indexes[node.leftFirst + i] );
        }
        return;
    }
    IntersectionData tData1;
    intersectBVH( ray, tData, node.leftFirst );
    intersectBVH( ray, tData1, node.leftFirst + 1 );
    if ( tData1.t < tData.t ) std::swap( tData1, tData );
}
