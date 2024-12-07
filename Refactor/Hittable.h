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

class Hittable {
public:

    Hittable(): material(nullptr) {}

    Hittable( Material* material ): material(material) {}

    Material* material;
};


#endif //COLLECTION_HITTABLE_H
