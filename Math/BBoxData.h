//
// Created by auser on 8/1/24.
//

#ifndef COLLECTION_BBOXDATA_H
#define COLLECTION_BBOXDATA_H
#include "Vector3f.h"

struct BBoxData {
    BBoxData( const Vector3f& pMin, const Vector3f& pMax ): pMin( pMin ), pMax( pMax ) {}
    Vector3f pMin;
    Vector3f pMax;
};

#endif //COLLECTION_BBOXDATA_H
