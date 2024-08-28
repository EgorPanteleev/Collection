//
// Created by auser on 5/12/24.
//

#ifndef COLLECTION_INTERSECTIONDATA_H
#define COLLECTION_INTERSECTIONDATA_H

#include "Vector3f.h"
#include "Primitive.h"
class Triangle;
class Sphere;
class IntersectionData {
public:
    IntersectionData();
    IntersectionData( float t, Primitive* prim );
    float t;
    Primitive* primitive;
};

#endif //COLLECTION_INTERSECTIONDATA_H
