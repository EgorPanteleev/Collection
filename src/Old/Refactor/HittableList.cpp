//
// Created by auser on 11/26/24.
//

#include "HittableList.h"

HittableList::HittableList(): hittables()  {

}

//void HittableList::add( Hittable* object ) {
//    objects.push_back( object );
//}

//[[nodiscard]] HOST_DEVICE bool HittableList::hit( const Ray& ray, const Interval<double>& interval, HitRecord& record ) const {
//    HitRecord tmpRecord;
//    bool hitAnything = false;
//    double closest = interval.max;
//    for ( auto object: objects ) {
//        if ( !object->hit( ray, { interval.min, closest }, tmpRecord ) ) continue;
//        hitAnything = true;
//        closest = tmpRecord.t;
//        record = tmpRecord;
//        record.material = object->material;
//    }
//    return hitAnything;
//}

void HittableList::clear() {
    hittables.clear();
}

//void copyToDevice(Type *host, Type *&device) {
//    HIP_ASSERT(hipMalloc(&device, sizeof(Type)));
//    HIP_ASSERT(hipMemcpy(device, host, sizeof(Type), hipMemcpyHostToDevice));
//}
//
//void copyToHost(Type *host, Type *device) {
//    HIP_ASSERT(hipMemcpy(host, device, sizeof(Type), hipMemcpyDeviceToHost));
//    HIP_ASSERT(hipFree(device));
//}


#if HIP_ENABLED
//HOST void HittableList::copyToDevice( HittableList*& device ) {
//    HIP_ASSERT(hipMalloc(&device, sizeof(HittableList)));
//    Vector<Hittable*>* tmp;
//    objects.copyToDevice<true>( tmp );
//    device->objects = std::move( *tmp );
//}
//
//HOST void HittableList::copyToHost( HittableList* host ) {
//
//}

#endif