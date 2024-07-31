//
// Created by igor on 08.01.2024.
//

#include "Ray.h"
#include <cmath>
__host__ __device__ Ray::Ray():origin(), direction(){}
__host__ __device__ Ray::Ray(const Vector3f& from, const Vector3f& dir):origin(from), direction( dir.normalize() ) {
}

__host__ __device__ Ray::~Ray() {
    origin = Vector3f();
    direction = Vector3f();
}