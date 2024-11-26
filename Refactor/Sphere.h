//
// Created by auser on 11/26/24.
//

#ifndef COLLECTION_SPHERE_H
#define COLLECTION_SPHERE_H

#include "Hittable.h"

class Sphere: public Hittable {
public:
    Sphere();
    Sphere( double r, const Point3d& pos, Material* mat );
    [[nodiscard]] bool hit( const Ray& ray, const Interval<double>& interval, HitRecord& record ) const override;

public:
    Point3d origin;
    double radius;
};

#endif //COLLECTION_SPHERE_H
