//
// Created by igor on 08.01.2024.
//

#include "Ray.h"
#include <cmath>
Ray::Ray():origin(), direction(){}
Ray::Ray(const Vector3f& from, const Vector3f& dir):origin(from), direction( dir.normalize() ) {
}

Vector3f Ray::getOrigin() const {
    return origin;
}
Vector3f Ray::getDirection() const {
    return direction;
}

void Ray::setOrigin( const Vector3f& orig ) {
    origin = orig;
}
void Ray::setDirection( const Vector3f& dir ) {
    direction = dir;
}

Ray::~Ray() {
    origin = Vector3f();
    direction = Vector3f();
}