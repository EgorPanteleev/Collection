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

    HOST void add( HittableList* hittableList )  {
        for ( auto hittable: hittableList->hittables )
            hittables.push_back( hittable );
    }

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

    BBox computeBBox() const {
        BBox bbox;
        for (auto& hittable : hittables ) {
            auto tri = static_cast<Triangle*>(hittable);
            bbox.merge( tri->v1 );
            bbox.merge( tri->v2 );
            bbox.merge( tri->v3 );
        }
        return bbox;
    }

    void translate( const Vec3d& translation ) {
        for (auto& hittable : hittables ) {
            auto tri = static_cast<Triangle*>(hittable);
            tri->v1 += translation;
            tri->v2 += translation;
            tri->v3 += translation;
        }
    }


    void translateTo( const Vec3d& target ) {
        BBox bbox = computeBBox();
        auto origin = ( bbox.pMin + bbox.pMax ) / 2;
        auto translation = target - origin;
        for (auto& hittable : hittables ) {
            auto tri = static_cast<Triangle*>(hittable);
            tri->v1 += translation;
            tri->v2 += translation;
            tri->v3 += translation;
        }
    }

    void scale( const Vec3d& scale ) {
        for (auto& hittable : hittables ) {
            auto tri = static_cast<Triangle*>(hittable);
            tri->v1 *= scale;
            tri->v2 *= scale;
            tri->v3 *= scale;
        }
    }

    void scaleTo( const Vec3d& scaleTo ) {
        BBox bbox = computeBBox();
        auto delta = bbox.pMax - bbox.pMin;
        auto scale = scaleTo / delta;
        for (auto& hittable : hittables ) {
            auto tri = static_cast<Triangle*>(hittable);
            tri->v1 *= scale;
            tri->v2 *= scale;
            tri->v3 *= scale;
        }
    }

    void clear();

#if HIP_ENABLED
    virtual HOST HittableList* copyToDevice() {
        auto objectsDevice = hittables.copyToDevice();
        auto device = HIP::allocateOnDevice<HittableList>();
        device->hittables = move(*objectsDevice);
        return nullptr;

    }

    virtual HOST HittableList* copyToHost() {
        auto host = new HittableList();
        HIP::copyToHost( host, this );

        auto hostObjects = hittables.copyToHost();
        host->hittables = move(*hostObjects);
        return host;
    }

    virtual HOST void deallocateOnDevice() {
        hittables.deallocateOnDevice();
        HIP::deallocateOnDevice<HittableList>( this );
    }
#endif
public:
    Vector<Hittable*> hittables;
};


#endif //COLLECTION_HITTABLELIST_H
