//
// Created by igor on 08.01.2024.
//

#include "Ray.h"
#include <cmath>

Ray::Ray():origin(), direction(), invDirection() {}

Ray::Ray(const Vec3d& from, const Vec3d& dir): origin(from), direction( dir.normalize() ) {
    invDirection = 1.0 / dir.normalize();
}

Ray::~Ray() {
    origin = Vec3d();
    direction = Vec3d();
}