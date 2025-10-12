//
// Created by auser on 1/2/25.
//

#include "BVH.h"

BVH::BVH(): indexes(), bvhNodes() {
}

void BVH::build() {
    //bvhNodes.resize( hittables.size() ); // TODO
    for (int i = 0; i < hittables.size(); ++i ) bvhNodes.push_back( {} );
    for (int i = 0; i < hittables.size(); ++i ) indexes.push_back( i );
    BVHNode& root = bvhNodes[rootNodeIdx];
    root.leftFirst = 0, root.count = hittables.size();
    Vec3d centroidMin, centroidMax;
    updateNodeBounds( 0, centroidMin, centroidMax );
    subDivide( 0, 0, nodesUsed, centroidMin, centroidMax );
}


void BVH::updateNodeBounds( uint nodeIdx, Vec3d& centroidMin, Vec3d& centroidMax ) {
    BVHNode& node = bvhNodes[nodeIdx];
    node.bbox = { INF, -INF };
    centroidMin = INF;
    centroidMax = -INF;
    for ( uint i = 0; i < node.count; ++i ) {
        BBox bbox = hittables[indexes[node.leftFirst + i]]->getBBox();
        node.bbox = { min( node.bbox.pMin, bbox.pMin ), max( node.bbox.pMax, bbox.pMax ) };
        centroidMin = min( centroidMin, bbox.getCentroid() );
        centroidMax = max( centroidMax, bbox.getCentroid() );
    }
}



void BVH::subDivide( uint nodeIdx, uint depth, uint& nodePtr, Vec3d& centroidMin, Vec3d& centroidMax ) {
    BVHNode& node = bvhNodes[nodeIdx];
    if (node.count <= 2) return;
//    if ( depth > 10000 ) return;
    int axis = 1, splitPos = 1;
    double splitCost = findBestSplitPlane( node, axis, splitPos, centroidMin, centroidMax );
    double nosplitCost = node.calculateNodeCost();
    if (splitCost >= nosplitCost) return;
    uint i = node.leftFirst;
    uint j = i + node.count - 1;
    double scale = BINS / (centroidMax[axis] - centroidMin[axis]);
    while (i <= j) {
        Vec3d origin = hittables[indexes[i]]->getBBox().getCentroid();
        int binIdx = std::min( BINS - 1, (int)((origin[axis] - centroidMin[axis]) * scale) );
        if (binIdx < splitPos) ++i; else std::swap( indexes[i], indexes[j--] );
    }
    uint leftCount = i - node.leftFirst;
    if (leftCount == 0 || leftCount == node.count) return; // never happens for dragon mesh, nice
    uint leftChildIdx = nodePtr++;
    uint rightChildIdx = nodePtr++;
    bvhNodes[leftChildIdx].leftFirst = node.leftFirst;
    bvhNodes[leftChildIdx].count = leftCount;
    bvhNodes[rightChildIdx].leftFirst = i;
    bvhNodes[rightChildIdx].count = node.count - leftCount;
    node.leftFirst = leftChildIdx;
    node.count = 0;
    updateNodeBounds( leftChildIdx, centroidMin, centroidMax );
    subDivide( leftChildIdx, depth + 1, nodePtr, centroidMin, centroidMax );
    updateNodeBounds( rightChildIdx, centroidMin, centroidMax );
    subDivide( rightChildIdx, depth + 1, nodePtr, centroidMin, centroidMax );
}


double BVH::findBestSplitPlane( BVHNode& node, int& axis, int& splitPos, Vec3d& centroidMin, Vec3d& centroidMax ) {
    double bestCost = INF;
    for ( int a = 0; a < 3; ++a ) {
        double boundsMin = centroidMin[a], boundsMax = centroidMax[a];
        if (boundsMin == boundsMax) continue;
        double scale = BINS / (boundsMax - boundsMin);
        double leftCountArea[BINS - 1], rightCountArea[BINS - 1];
        int leftSum = 0, rightSum = 0;
        Bin bin[BINS];
        for ( uint i = 0; i < node.count; ++i ) {
            BBox bbox = hittables[indexes[node.leftFirst + i]]->getBBox();
            int binIdx = std::min( BINS - 1, (int)((bbox.getCentroid()[a] - boundsMin) * scale) );
            bin[binIdx].count++;
            bin[binIdx].bounds.merge( bbox );
        }
        BBox leftBox, rightBox;
        for (int i = 0; i < BINS - 1; i++) {
            leftSum += bin[i].count;
            leftBox.merge( bin[i].bounds );
            leftCountArea[i] = leftSum * leftBox.getArea();
            rightSum += bin[BINS - 1 - i].count;
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
        std::cout << "ind - "<< ind << std::endl;
        std::cout << "count - "<< node.count << std::endl;
        std::cout << "left first - "<< node.leftFirst << std::endl;
        std::cout << "bbox - "<< node.bbox << std::endl;
    }
}

#if HIP_ENABLED
HOST HittableList* BVH::copyToDevice() {
    auto device = HIP::allocateOnDevice<BVH>();

    device->hittables = move(*hittables.copyToDevice());
    device->indexes = move(*indexes.copyToDevice());
    device->bvhNodes = move(*bvhNodes.copyToDevice());
    return device;
}

HOST HittableList* BVH::copyToHost() {
    auto host = new BVH();
    HIP::copyToHost( host, this );

    host->hittables = move(*hittables.copyToHost());
    host->indexes = move(*indexes.copyToHost());
    host->bvhNodes = move(*bvhNodes.copyToHost());
    return host;
}

HOST void BVH::deallocateOnDevice() {
    hittables.deallocateOnDevice();
    indexes.deallocateOnDevice();
    bvhNodes.deallocateOnDevice();

    HIP::deallocateOnDevice<BVH>( this );
}
#endif

