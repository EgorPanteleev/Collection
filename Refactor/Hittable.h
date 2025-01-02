//
// Created by auser on 11/26/24.
//

#ifndef COLLECTION_HITTABLE_H
#define COLLECTION_HITTABLE_H

#include "Vec3.h"
#include "Ray.h"
#include "Interval.h"
#include "HitRecord.h"
#include "Material.h"
#include "SystemUtils.h"
#include "BBox.h"

class Hittable {
public:
    enum Type {
        SPHERE,
        UNKNOWN
    };

    Hittable();

    Hittable( Type type );

    Hittable( Material* material );

    Hittable( Type type, Material* material );

    BBox getBBox() const;

#if HIP_ENABLED
    virtual HOST Hittable* copyToDevice() = 0;

    virtual HOST Hittable* copyToHost() = 0;

    virtual HOST void deallocateOnDevice() = 0;
#endif

    Type type;

    BBox bbox;

    Material* material;
};


#endif //COLLECTION_HITTABLE_H
