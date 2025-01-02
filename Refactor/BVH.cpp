//
// Created by auser on 1/2/25.
//

#include "BVH.h"

BVH::BVH(): indexes(), bvhNodes() {
}

void BVH::build() {
    bvhNodes.resize( hittables.size() );
    //for (int i = 0; i < hittables.size(); ++i ) bvhNodes.push_back( {} );
    for (int i = 0; i < hittables.size(); ++i ) indexes.push_back( i );
    BVHNode& root = bvhNodes[rootNodeIdx];
    root.leftFirst = 0, root.trianglesCount = hittables.size();
    Vec3d centroidMin, centroidMax;
    updateNodeBounds( 0, centroidMin, centroidMax );
    subDivide( 0, 0, nodesUsed, centroidMin, centroidMax );
}

void BVH::updateNodeBounds( uint nodeIdx, Vec3d& centroidMin, Vec3d& centroidMax ) {
    BVHNode& node = bvhNodes[nodeIdx];
    node.bbox = { 1e30, -1e30 };
    centroidMin = 1e30;
    centroidMax = -1e30;
    for (uint first = node.leftFirst, i = 0; i < node.trianglesCount; ++i) {
        BBox bbox = hittables[indexes[first + i]]->getBBox();
        node.bbox = { min( node.bbox.pMin, bbox.pMin ), max( node.bbox.pMax, bbox.pMax ) };
        centroidMin = min( centroidMin, hittables[indexes[first + i]]->bbox.getCentroid() );
        centroidMax = max( centroidMax, hittables[indexes[first + i]]->bbox.getCentroid() );
    }
}

void BVH::subDivide( uint nodeIdx, uint depth, uint& nodePtr, Vec3d& centroidMin, Vec3d& centroidMax ) {
    BVHNode& node = bvhNodes[nodeIdx];
    if (node.trianglesCount <= 2) return;
    int axis = 1, splitPos = 1;
    double splitCost = findBestSplitPlane( node, axis, splitPos, centroidMin, centroidMax );
    double nosplitCost = node.calculateNodeCost();
    if (splitCost >= nosplitCost) return;
    int i = node.leftFirst;
    int j = i + node.trianglesCount - 1;
    double scale = BINS / (centroidMax[axis] - centroidMin[axis]);
    while (i <= j) {
        Vec3d origin = hittables[indexes[i]]->getBBox().getCentroid();
        int binIdx = std::min( BINS - 1, (int)((origin[axis] - centroidMin[axis]) * scale) );
        if (binIdx < splitPos) i++; else std::swap( indexes[i], indexes[j--] );
    }
    int leftCount = i - node.leftFirst;
    if (leftCount == 0 || leftCount == node.trianglesCount) return; // never happens for dragon mesh, nice
    int leftChildIdx = nodePtr++;
    int rightChildIdx = nodePtr++;
    bvhNodes[leftChildIdx].leftFirst = node.leftFirst;
    bvhNodes[leftChildIdx].trianglesCount = leftCount;
    bvhNodes[rightChildIdx].leftFirst = i;
    bvhNodes[rightChildIdx].trianglesCount = node.trianglesCount - leftCount;
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
            int binIdx =std:: min( BINS - 1, (int)((hittables[indexes[node.leftFirst + i]]->bbox.getCentroid()[a] - boundsMin) * scale) );
            bin[binIdx].trianglesCount++;
            BBox bbox = hittables[indexes[node.leftFirst + i]]->getBBox();
            bin[binIdx].bounds.merge( bbox );
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

void BVH::printDebug() const {
    for ( auto ind: indexes ) {
        auto node = bvhNodes[ind];
        std::cout << "ind - "<< node.trianglesCount << std::endl;
        std::cout << "triangles count - "<< node.trianglesCount << std::endl;
        std::cout << "left first - "<< node.leftFirst << std::endl;
        std::cout << "bbox - "<< node.bbox << std::endl;
    }
}

#if HIP_ENABLED
HOST HittableList* BVH::copyToDevice() {
    auto hittablesDevice = hittables.copyToDevice();
    auto indexesDevice = indexes.copyToDevice();
    auto nodesDevice = bvhNodes.copyToDevice();

    auto device = HIP::allocateOnDevice<BVH>();

    std::swap( device->hittables, *hittablesDevice );
    std::swap( device->indexes, *indexesDevice );
    std::swap( device->bvhNodes, *nodesDevice );
    return device;
}

HOST HittableList* BVH::copyToHost() {
    auto host = new BVH();
    HIP::copyToHost( host, this );

    auto hostHittables = hittables.copyToHost();
    auto hostIndexes = indexes.copyToHost();
    auto hostNodes = bvhNodes.copyToHost();

    std::swap( host->hittables, *hostHittables );
    std::swap( host->indexes, *hostIndexes );
    std::swap( host->bvhNodes, *hostNodes );
    return host;
}

HOST void BVH::deallocateOnDevice() {
    hittables.deallocateOnDevice();
    indexes.deallocateOnDevice();
    bvhNodes.deallocateOnDevice();

    //TODO  HIP::deallocateOnDevice<HittableList>( this );
}
#endif

