//
// Created by auser on 5/12/24.
//

#ifndef COLLECTION_INTERSECTIONDATA_H
#define COLLECTION_INTERSECTIONDATA_H

#include "Vector3f.h"
#include "Triangle.h"
class Triangle;
class IntersectionData {
public:
    __host__ __device__ IntersectionData();
    __host__ __device__ IntersectionData( float t, const Vector3f& N, Triangle* tr );
    float t;
    Vector3f N;
    Triangle* triangle;
};

#endif //COLLECTION_INTERSECTIONDATA_H
