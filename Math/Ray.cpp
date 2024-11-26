//
// Created by igor on 08.01.2024.
//

#include "Ray.h"
#include <cmath>

Ray::Ray():origin(), direction() {}

Ray::Ray(const Vec3d& from, const Vec3d& dir): origin(from), direction( dir.normalize() ) {
}

Ray::~Ray() {
    origin = Vec3d();
    direction = Vec3d();
}

Vec3d Ray::at( double t ) const {
    return origin + t * direction;
}