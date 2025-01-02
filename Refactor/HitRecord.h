//
// Created by auser on 12/1/24.
//

#ifndef COLLECTION_HITRECORD_H
#define COLLECTION_HITRECORD_H

#include "Vec3.h"
#include "Ray.h"

class Material;

class HitRecord {
public:
    HitRecord() = default;

    Material* material;
    Point3d p;
    Vec3d N;
    double t;
    bool frontFace;

    HOST_DEVICE void setFaceNormal( const Ray& ray, const Vec3d& outwardNormal ) {
        frontFace = dot( ray.direction, outwardNormal ) < 0;
        N = frontFace ? outwardNormal : -outwardNormal;
    }
};


#endif //COLLECTION_HITRECORD_H
