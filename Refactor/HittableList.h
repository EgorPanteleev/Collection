//
// Created by auser on 11/26/24.
//

#ifndef COLLECTION_HITTABLELIST_H
#define COLLECTION_HITTABLELIST_H

#include "Vector.h"
#include "Hittable.h"

class HittableList {
public:
    HittableList();

    void add( Hittable* object );

    [[nodiscard]] bool hit( const Ray& ray, const Interval<double>& interval, HitRecord& record ) const;

    void clear();
private:
    Vector<Hittable*> objects;
};


#endif //COLLECTION_HITTABLELIST_H
