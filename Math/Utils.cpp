#include "Utils.h"
#include <cmath>
double dot( Vector3f p1, Vector3f p2 ) {
    return ( p1.getX() * p2.getX() + p1.getY() * p2.getY() + p1.getZ() * p2.getZ());
}

Vector3f normalize( Vector3f p ) {
    double lenght = sqrt( pow( p.getX(), 2 ) +  pow( p.getY(), 2 ) + pow( p.getZ(), 2 ));
    Vector3f p1;
    p1 = p / lenght;
    return p1;
}