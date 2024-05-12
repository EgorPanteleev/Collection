//
// Created by auser on 5/12/24.
//

#ifndef COLLECTION_INTERSECTIONDATA_H
#define COLLECTION_INTERSECTIONDATA_H

#include "Vector3f.h"
#include "Object.h"
class Object;
class IntersectionData {
public:
    IntersectionData();
    IntersectionData( float t, const Vector3f& N, Object* obj );
    float t;
    Vector3f N;
    Object* object;
};

#endif //COLLECTION_INTERSECTIONDATA_H
