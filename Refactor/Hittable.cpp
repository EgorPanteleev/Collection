//
// Created by auser on 11/26/24.
//

#include "Hittable.h"

Hittable::Hittable(): material(nullptr), type(UNKNOWN), bbox() {}

Hittable::Hittable( Type type ): material(nullptr), type(type), bbox() {}

Hittable::Hittable( Material* material ): type(UNKNOWN), material(material), bbox() {}

Hittable::Hittable( Type type, Material* material ): material(material), type(type), bbox() {}

BBox Hittable::getBBox() const {
    return bbox;
}