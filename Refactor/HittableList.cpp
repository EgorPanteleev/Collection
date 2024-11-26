//
// Created by auser on 11/26/24.
//

#include "HittableList.h"

HittableList::HittableList(): objects()  {

}

void HittableList::add( Hittable* object ) {
    objects.push_back( object );
}

[[nodiscard]] bool HittableList::hit( const Ray& ray, const Interval<double>& interval, HitRecord& record ) const {
    HitRecord tmpRecord;
    bool hitAnything = false;
    double closest = interval.max;
    for ( auto object: objects ) {
        if ( !object->hit( ray, { interval.min, closest }, tmpRecord ) ) continue;
        hitAnything = true;
        closest = tmpRecord.t;
        record = tmpRecord;
        record.material = object->material;
    }
    return hitAnything;
}

void HittableList::clear() {
    objects.clear();
}