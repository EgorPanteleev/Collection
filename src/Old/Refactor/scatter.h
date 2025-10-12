//
// Created by auser on 12/7/24.
//

#ifndef COLLECTION_SCATTER_H
#define COLLECTION_SCATTER_H

#include "Material.h"
#include "Sphere.h"
#include "Triangle.h"

DEVICE bool hit( Hittable* hittable, const Ray& ray, const Interval<double>& interval, HitRecord& record ) {
    switch (hittable->type) {
        case Hittable::SPHERE: {
            return static_cast<Sphere*>(hittable)->hit( ray, interval, record );
        }
        case Hittable::TRIANGLE: {
            return static_cast<Triangle*>(hittable)->hit( ray, interval, record );
        }
        default: {
            return false;
        }
    }
}

DEVICE bool scatter( Material* material, const Ray& rayIn, const HitRecord& hitRecord, RGB& attenuation, Ray& scattered, hiprandState& state ) {
    switch (material->type) {
        case Material::LAMBERTIAN: {
            return static_cast<Lambertian*>(material)->scatter( rayIn, hitRecord, attenuation, scattered, state );
        }
        case Material::ORENNAYAR: {
            return static_cast<OrenNayar*>(material)->scatter( rayIn, hitRecord, attenuation, scattered, state );
        }
        case Material::METAL: {
            return static_cast<Metal*>(material)->scatter( rayIn, hitRecord, attenuation, scattered, state );
        }
        case Material::DIELECTRIC:{
            return static_cast<Dielectric*>(material)->scatter( rayIn, hitRecord, attenuation, scattered, state );
        }
        default: {
            return false;
        }
    }
}

DEVICE RGB emit( Material* material, double u, double v, const Point3d & p) {
    switch (material->type) {
        case Material::LIGHT: {
            return static_cast<Light*>(material)->emit( u, v, p );
        }
        default: {
            return { 0, 0, 0 };
        }
    }
}


#endif //COLLECTION_SCATTER_H
