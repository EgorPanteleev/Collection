//
// Created by auser on 11/26/24.
//

#include "Sphere.h"

//Sphere::Sphere(): radius() {
//}
//Sphere::Sphere( double r, const Point3d& pos, Material* mat ): radius( r ), origin(pos) {
//    material = mat;
//}

//HOST_DEVICE bool Sphere::hit( const Ray& ray, const Interval<double>& interval, HitRecord& record ) const {
//    Vec3d D = ray.direction;
//    Vec3d OC = ray.origin - origin;
//    double k1 = dot( D, D );
//    double k2 = 2 * dot( OC, D );
//    double k3 = dot( OC, OC ) - radius * radius;
//
//    double disc = k2 * k2 - 4 * k1 * k3;
//    if ( disc < 0 ) {
//        return false;
//    }
//    disc = sqrt( disc ) / ( 2 * k1 );
//    k2 = -k2 / ( 2 * k1 );
//    double t1 = k2 + disc;
//    double t2 = k2 - disc;
//    if ( t1 < t2 ) t2 = t1;
//
//    if ( !interval.contains( t2 ) ) return false;
//    record.t = t2;
//    record.p = ray.at( t2 );
//    record.setFaceNormal( ray, ( record.p - origin ) / radius );
//
//    return true;
//}
