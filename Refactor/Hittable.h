//
// Created by auser on 11/26/24.
//

#ifndef COLLECTION_HITTABLE_H
#define COLLECTION_HITTABLE_H

#include "Vec3.h"
#include "Ray.h"
#include "Interval.h"

class Material;

class HitRecord {
public:
    Material* material;
    Point3d p;
    Vec3d N;
    double t;
    bool frontFace;

    void setFaceNormal( const Ray& ray, const Vec3d& outwardNormal ) {
        frontFace = dot( ray.direction, outwardNormal ) < 0;
        N = frontFace ? outwardNormal : -outwardNormal;
    }
};

class Hittable {
public:
    [[nodiscard]] virtual bool hit( const Ray& ray, const Interval<double>& interval, HitRecord& record ) const = 0;
    Material* material;
};


#endif //COLLECTION_HITTABLE_H
