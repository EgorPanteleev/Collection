//
// Created by auser on 11/26/24.
//

#ifndef COLLECTION_HITTABLELIST_H
#define COLLECTION_HITTABLELIST_H

#include "Vector.h"
#include "Sphere.h"
#include "SystemUtils.h"
#include "scatter.h"

#if HIP_ENABLED
    #include "hip/hip_runtime.h"
#endif


class HittableList {
public:
    HittableList();

    HOST void add( Hittable* hittable )  {
        hittables.push_back( hittable );
    }

    [[nodiscard]] HOST_DEVICE bool hit( const Ray& ray, const Interval<double>& interval, HitRecord& record ) const {
        HitRecord tmpRecord;
        bool hitAnything = false;
        double closest = interval.max;
        for ( auto hittable: hittables ) {
            if ( !::hit( hittable, ray, { interval.min, closest }, tmpRecord ) ) continue;
            hitAnything = true;
            closest = tmpRecord.t;
            record = tmpRecord;
            record.material = hittable->material;
        }
        return hitAnything;
    }

    void clear();

#if HIP_ENABLED
    virtual HOST HittableList* copyToDevice() {
        auto objectsDevice = hittables.copyToDevice();

        auto device = HIP::allocateOnDevice<HittableList>();

        std::swap( device->hittables, *objectsDevice );
        return device;
    }

    virtual HOST HittableList* copyToHost() {
        auto host = new HittableList();
        HIP::copyToHost( host, this );

        auto hostObjects = hittables.copyToHost();
//        objects.swap( *hostObjects );
        std::swap( host->hittables, *hostObjects );
        return host;
    }

    virtual HOST void deallocateOnDevice() {
        hittables.deallocateOnDevice();

      //TODO  HIP::deallocateOnDevice<HittableList>( this );
    }
#endif
public:
    Vector<Hittable*> hittables;
};


#endif //COLLECTION_HITTABLELIST_H
