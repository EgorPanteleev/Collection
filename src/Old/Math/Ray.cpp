//
// Created by igor on 08.01.2024.
//

#include "Ray.h"
#include <cmath>

//HOST_DEVICE Ray::Ray():origin(), direction() {}
//
//HOST_DEVICE Ray::Ray(const Vec3d& from, const Vec3d& dir): origin(from), direction( dir.normalize() ) {
//}
//
//HOST_DEVICE Ray::~Ray() {
//    origin = Vec3d();
//    direction = Vec3d();
//}
//
//HOST_DEVICE Vec3d Ray::at( double t ) const {
//    return origin + t * direction;
//}