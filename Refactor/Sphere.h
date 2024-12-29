//
// Created by auser on 11/26/24.
//

#ifndef COLLECTION_SPHERE_H
#define COLLECTION_SPHERE_H

#include "Hittable.h"

class Sphere: public Hittable {
public:
    HOST_DEVICE Sphere() {}

    HOST_DEVICE Sphere( double r, const Point3d& pos, Material* mat ): Hittable(mat), radius( r ), origin(pos) {
    }
    [[nodiscard]] HOST_DEVICE bool hit( const Ray& ray, const Interval<double>& interval, HitRecord& record ) const {
       // printf("HI SPHERE::hit\n");
        Vec3d D = ray.direction;
        Vec3d OC = ray.origin - origin;
        double k1 = dot( D, D );
        double k2 = 2 * dot( OC, D );
        double k3 = dot( OC, OC ) - radius * radius;

        double disc = k2 * k2 - 4 * k1 * k3;
        if ( disc < 0 ) {
            return false;
        }
        disc = sqrt( disc ) / ( 2 * k1 );
        k2 = -k2 / ( 2 * k1 );
        double t1 = k2 + disc;
        double t2 = k2 - disc;
        if ( t1 < t2 ) t2 = t1;

        if ( !interval.contains( t2 ) ) return false;
        record.t = t2;
        record.p = ray.at( t2 );
        record.setFaceNormal( ray, ( record.p - origin ) / radius );

        return true;
    }


#if HIP_ENABLED
    Sphere* copyToDevice() {
        auto deviceMaterial = material->copyToDevice();
        auto originalMaterial = material;
        material = deviceMaterial;

        auto deviceSphere = HIP::allocateOnDevice<Sphere>();

        HIP::copyToDevice( this, deviceSphere );

        material = originalMaterial;
        return deviceSphere;

    }

    Sphere* copyToHost() {
        auto host = new Sphere();
        HIP::copyToHost( host, this );

        auto hostMaterial = material->copyToHost();
        host->material = hostMaterial;
        return host;
    }

    void deallocateOnDevice() {
        material->deallocateOnDevice();
        HIP::deallocateOnDevice( this );
    }
#endif
public:
    Point3d origin;
    double radius;
};


#endif //COLLECTION_SPHERE_H
