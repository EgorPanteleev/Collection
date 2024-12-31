//
// Created by auser on 11/26/24.
//

#ifndef COLLECTION_HITTABLELIST_H
#define COLLECTION_HITTABLELIST_H

#include "Vector.h"
#include "Sphere.h"
#include "SystemUtils.h"

#if HIP_ENABLED
    #include "hip/hip_runtime.h"
#endif

template <typename Type>
HOST_DEVICE bool hit( Type hittable, const Ray& ray, const Interval<double>& interval, HitRecord& record ) {
    return hittable->hit( ray, interval, record );
}


class HittableList {
public:
    HittableList();

    HOST_DEVICE void add( Sphere* object )  {
        spheres.push_back( object );
    }

    [[nodiscard]] HOST_DEVICE bool hit( const Ray& ray, const Interval<double>& interval, HitRecord& record ) const {
        HitRecord tmpRecord;
        bool hitAnything = false;
        double closest = interval.max;
        for ( auto sphere: spheres ) {
            if ( !::hit( sphere, ray, { interval.min, closest }, tmpRecord ) ) continue;
            hitAnything = true;
            closest = tmpRecord.t;
            record = tmpRecord;
            record.material = sphere->material;
        }
        return hitAnything;
    }

    void clear();

#if HIP_ENABLED
    HOST HittableList* copyToDevice() {
        auto objectsDevice = spheres.copyToDevice();

        auto device = HIP::allocateOnDevice<HittableList>();

        //device->spheres.swap( *objectsDevice );

        std::swap( device->spheres, *objectsDevice );

//        auto objectsDevice = objects.copyToDevice<true>();
//        auto device = HIP::allocateOnDevice<HittableList>();
//        auto originalObjects = &objects;
//        objects.swap( *objectsDevice );
//        HIP::copyToDevice( this, device );
//
//        objects.swap( *originalObjects );
        return device;
    }

    HOST HittableList* copyToHost() {
        auto host = new HittableList();
        HIP::copyToHost( host, this );

        auto hostObjects = spheres.copyToHost();
//        objects.swap( *hostObjects );
        std::swap( host->spheres, *hostObjects );
        return host;
    }

    HOST void deallocateOnDevice() {
        spheres.deallocateOnDevice();

      //TODO  HIP::deallocateOnDevice<HittableList>( this );
    }
#endif
public:
    Vector<Sphere*> spheres;
};


#endif //COLLECTION_HITTABLELIST_H
