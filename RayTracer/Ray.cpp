//
// Created by igor on 08.01.2024.
//

#include "Ray.h"
#include "cmath"
Ray::Ray():origin(), direction(){}
Ray::Ray(Point from, Point to):origin(from) {
    direction = to - from;
    double lenght = sqrt( pow( direction.getX(), 2 ) +  pow( direction.getY(), 2 ) + pow( direction.getZ(), 2 ));
    direction = direction / lenght;
}

Point Ray::getOrigin() const {
    return origin;
}
Point Ray::getDirection() const {
    return direction;
}

void Ray::setOrigin( Point orig ) {
    origin = orig;
}
void Ray::setDirection( Point dir ) {
    direction = dir;
}

Ray::~Ray() {
    origin = Point();
    direction = Point();
}