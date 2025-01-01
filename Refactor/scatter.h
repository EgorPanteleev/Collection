//
// Created by auser on 12/7/24.
//

#ifndef COLLECTION_SCATTER_H
#define COLLECTION_SCATTER_H

#include "Material.h"
#include "Sphere.h"

DEVICE bool hit( Hittable* hittable, const Ray& ray, const Interval<double>& interval, HitRecord& record ) {
    switch (hittable->type) {
        case Hittable::SPHERE: {
            return ((Sphere*) hittable)->hit( ray, interval, record );
        }
        default: {
            return false;
        }
    }
}

DEVICE bool scatter( Material* material, const Ray& rayIn, const HitRecord& hitRecord, RGB& attenuation, Ray& scattered, hiprandState& state ) {
    switch (material->type) {
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
