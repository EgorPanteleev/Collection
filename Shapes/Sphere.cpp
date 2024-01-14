#include <cmath>
#include "Sphere.h"
double dot( Point& p1, Point& p2 ) {
    return ( p1.getX() * p2.getX() + p1.getY() * p2.getY() + p1.getZ() * p2.getZ());
}
Sphere::Sphere(): radius(0), origin() {
}
Sphere::Sphere( double r, Point pos, RGB _color ): radius(r), origin(pos) {
    setColor( _color );
}

double Sphere::intersectsWithRay( const Ray& ray ) {
    Point D = ray.getDirection();
    Point OC = ray.getOrigin() - origin;
    double k1 = dot( D, D );
    double k2 = 2 * dot( OC, D );
    double k3 = dot( OC, OC ) - radius * radius;

    int disc = k2 * k2 - 4 * k1 * k3;
    if ( disc < 0 ) {
        return std::numeric_limits<double>::min();
    }
    double t1 = (-k2 + sqrt(disc)) / (2 * k1);
    double t2 = (-k2 - sqrt(disc)) / (2 * k1);
    if ( t1 < t2 ) t2 = t1;
    return t2;
}

Point Sphere::getNormal( Point p ) {
    return ( p - origin );
}