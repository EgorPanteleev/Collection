//
// Created by auser on 1/2/25.
//

#ifndef COLLECTION_BVH_H
#define COLLECTION_BVH_H
#define BINS 8
#include <cmath>
#include "Vector.h"
#include "Sphere.h"
#include "HittableList.h"

class BVHNode {
public:
    BVHNode() = default;

    [[nodiscard]] HOST_DEVICE bool isLeaf() const {
        return ( count > 0 );
    }
    [[nodiscard]] double calculateNodeCost() const {
        Vec3d e = bbox.pMax - bbox.pMin;
        return (e[0] * e[1] + e[1] * e[2] + e[2] * e[0]) * count;
    }

    BBox bbox;
    uint leftFirst, count;
};

class Bin {
public:
    Bin(): count(0), bounds() {}
    BBox bounds;
    int count;
};

//TODO try to store vector of hittable lists?

class BVH: public HittableList {
public:

    BVH();

    void build();

    void updateNodeBounds( uint nodeIdx, Vec3d& centroidMin, Vec3d& centroidMax );

    void subDivide( uint nodeIdx, uint depth, uint& nodePtr, Vec3d& centroidMin, Vec3d& centroidMax );

    double findBestSplitPlane( BVHNode& node, int& axis, int& splitPos, Vec3d& centroidMin, Vec3d& centroidMax );

    DEVICE bool hit( const Ray& ray, const Interval<double>& interval, HitRecord& record ) const {
        Vector<const BVHNode*> stack; //TODO stack
        stack.push_back( &bvhNodes[0] );
        bool hitAnything = false;
        while ( !stack.empty() ) {
            auto node = stack.back();
            stack.pop_back();
            if ( !node->bbox.intersectsWithRay( ray ) ) {
                continue;
            }
            if (node->isLeaf()) {
                HitRecord tmpRecord;
                for (uint i = 0; i < node->count; ++i ) {
                    auto hittable = hittables[indexes[node->leftFirst + i]];
                    if ( !::hit(hittable, ray, interval, tmpRecord ) ) continue;
                    hitAnything = true;
                    if ( tmpRecord.t >= record.t ) continue;
                    record = tmpRecord;
                    record.material = hittable->material;
                }
            } else {
                stack.push_back( &bvhNodes[ node->leftFirst ] );
                stack.push_back( &bvhNodes[ node->leftFirst + 1 ] );
            }
        }
        return hitAnything;
    }

//    DEVICE bool hit( const Ray& ray, const Interval<double>& interval, HitRecord& record, uint nodeIdx ) const {
//        const BVHNode& node = bvhNodes[nodeIdx];
//        if ( !node.bbox.intersectsWithRay( ray ) ) {
//            return false;
//        }
//
//
//        if (node.isLeaf()) {
//            HitRecord tmpRecord;
//            //tmpRecord.t = interval.max;
//            bool hitAnything = false;
//            for (uint i = 0; i < node.trianglesCount; ++i ) {
//                auto hittable = hittables[indexes[node.leftFirst + i]];
//                if ( !::hit(hittable, ray, interval, tmpRecord ) ) continue;
//                hitAnything = true;
//                if ( tmpRecord.t >= record.t ) continue;
//                record = tmpRecord;
//                record.material = hittable->material;
//            }
//            return hitAnything;
//        }
//        HitRecord tmpRecord;
//        //tmpRecord.t = interval.max;
//        bool status1 = hit( ray, interval, record, node.leftFirst );
//        bool status2 = hit( ray, interval, tmpRecord, node.leftFirst + 1 );
//        if ( tmpRecord.t < record.t && status1 || status2 ) {
//            HitRecord tmp;
//            tmp = record;
//            record = tmpRecord;
//            tmpRecord = tmp;
//        }
//        return status1 || status2;
//    }

    void printDebug() const;

#if HIP_ENABLED
    HOST HittableList* copyToDevice() override;

    HOST HittableList* copyToHost() override;

    HOST void deallocateOnDevice() override;
#endif

public:
    Vector <uint> indexes;
    Vector<BVHNode> bvhNodes;
    uint rootNodeIdx = 0, nodesUsed = 1;
};



#endif //COLLECTION_BVH_H
