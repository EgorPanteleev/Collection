//
// Created by auser on 5/12/24.
//

#ifndef COLLECTION_INTERSECTIONDATA_H
#define COLLECTION_INTERSECTIONDATA_H

#include "Vector3f.h"
#include "Triangle.h"
#include "Sphere.h"
class Triangle;
class Sphere;
class IntersectionData {
public:
    IntersectionData();
    IntersectionData( float t, const Vector3f& N, Triangle* tr, Sphere* sp );
    float t;
    Vector3f N;
    Sphere* sphere;
    Triangle* triangle;
};

#endif //COLLECTION_INTERSECTIONDATA_H
