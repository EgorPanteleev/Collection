//
// Created by igor on 08.01.2024.
//

#include "Ray.h"

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