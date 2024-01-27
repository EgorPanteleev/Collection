#include <cmath>
#include "Sphere.h"
#include "Utils.h"
Sphere::Sphere(): radius(0), origin() {
}
Sphere::Sphere( double r, const Vector3f& pos, const RGB& _color ): radius(r), origin(pos) {
    setColor( _color );
}

bool Sphere::isContainPoint( const Vector3f& p ) const {
    if ( getDistance( p, origin ) == radius ) return true;
    return false;
}

float Sphere::intersectsWithRay( const Ray& ray ) const {
    Vector3f D = ray.getDirection();
    Vector3f OC = ray.getOrigin() - origin;
    float k1 = dot( D, D );
    float k2 = 2 * dot( OC, D );
    float k3 = dot( OC, OC ) - radius * radius;

    int disc = k2 * k2 - 4 * k1 * k3;
    if ( disc < 0 ) {
        return std::numeric_limits<float>::min();
    }
    float t1 = (-k2 + sqrt(disc)) / (2 * k1);
    float t2 = (-k2 - sqrt(disc)) / (2 * k1);
    if ( t1 < t2 ) t2 = t1;
    return t2;
}

Vector3f Sphere::getNormal( const Vector3f& p ) const {
    return ( p - origin );
}