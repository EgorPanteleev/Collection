#include <cmath>
#include "Sphere.h"
#include "Utils.h"
Sphere::Sphere(): radius(0), origin() {
}
Sphere::Sphere( float r, const Vector3f& pos, const RGB& color ): radius(r), origin(pos) {
    setColor( color );
}

bool Sphere::isContainPoint( Vector3f p ) const {
    if ( getDistance( p, origin ) == radius ) return true;
    return false;
}

float Sphere::intersectsWithRay( const Ray& ray ) const {
    Vector3f D = ray.getDirection();
    Vector3f OC = ray.getOrigin() - origin;
    float k1 = dot( D, D );
    float k2 = 2 * dot( OC, D );
    float k3 = dot( OC, OC ) - radius * radius;

    float disc = k2 * k2 - 4 * k1 * k3;
    if ( disc < 0 ) return std::numeric_limits<double>::min();
    float t1 = (-k2 + std::sqrt(disc)) / (2 * k1);
    float t2 = (-k2 - std::sqrt(disc)) / (2 * k1);
    return ( t1 < t2 ) ? t1 : t2;
}

Vector3f Sphere::getNormal( Vector3f p ) const {
    return ( p - origin );
}