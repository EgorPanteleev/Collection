//
// Created by auser on 11/26/24.
//

#include "Hittable.h"

Hittable::Hittable(): material(nullptr), type(UNKNOWN) {}

Hittable::Hittable( Type type ): material(nullptr), type(type) {}

Hittable::Hittable( Material* material ): type(UNKNOWN), material(material) {}

Hittable::Hittable( Type type, Material* material ): material(material), type(type) {}