//
// Created by auser on 12/7/24.
//

#ifndef COLLECTION_SCATTER_H
#define COLLECTION_SCATTER_H

#include "Material.h"

DEVICE bool scatter( Material* material, const Ray& rayIn, const HitRecord& hitRecord, RGB& attenuation, Ray& scattered, hiprandState& state ) {
    auto type = material->type;
    switch (type) {
        case Material::LAMBERTIAN: {
            return ((Lambertian*) material)->scatter( rayIn, hitRecord, attenuation, scattered, state );
        }
        case Material::METAL: {
            return ((Metal*) material)->scatter( rayIn, hitRecord, attenuation, scattered, state );
        }
        case Material::DIELECTRIC:{
            return ((Dielectric*) material)->scatter( rayIn, hitRecord, attenuation, scattered, state );
        }
        default: {
            return false;
        }
    }
}


#endif //COLLECTION_SCATTER_H
