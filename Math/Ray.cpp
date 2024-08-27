//
// Created by igor on 08.01.2024.
//

#include "Ray.h"
#include <cmath>
#include "Utils.h"
Ray::Ray():origin(), direction(), invDirection() {}
Ray::Ray(const Vector3f& from, const Vector3f& dir): origin(from), direction( dir.normalize() ) {
    invDirection = 1.0f / dir.normalize();
}

Ray::~Ray() {
    origin = Vector3f();
    direction = Vector3f();
}