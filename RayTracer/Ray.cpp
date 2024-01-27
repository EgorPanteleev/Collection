//
// Created by igor on 08.01.2024.
//

#include "Ray.h"
#include <cmath>
Ray::Ray():origin(), direction(){}
Ray::Ray(Vector3f from, Vector3f to):origin(from) {
    direction = to - from;
    double lenght = sqrt( pow( direction.getX(), 2 ) +  pow( direction.getY(), 2 ) + pow( direction.getZ(), 2 ));
    direction = direction / lenght;
}

Vector3f Ray::getOrigin() const {
    return origin;
}
Vector3f Ray::getDirection() const {
    return direction;
}

void Ray::setOrigin( Vector3f orig ) {
    origin = orig;
}
void Ray::setDirection( Vector3f dir ) {
    direction = dir;
}

Ray::~Ray() {
    origin = Vector3f();
    direction = Vector3f();
}