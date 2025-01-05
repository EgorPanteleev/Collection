//
// Created by auser on 1/5/25.
//

#ifndef COLLECTION_TRIANGLE_H
#define COLLECTION_TRIANGLE_H

#include "BBox.h"
#include "Ray.h"
#include "HitRecord.h"
#include "Interval.h"
#include "Hittable.h"

class Triangle: public Hittable {
public:
    Triangle();
    Triangle( const Vec3d& v1, const Vec3d& v2, const Vec3d& v3 );
    Triangle( const Vec3d& v1, const Vec3d& v2, const Vec3d& v3, Material* mat );
    [[nodiscard]] Vec3d getNormal() const;
    DEVICE bool hit( const Ray& ray, const Interval<double>& interval, HitRecord& record ) const  {
        Vec3d edge1 = v2 - v1;
        Vec3d edge2 = v3 - v1;
        Vec3d h = cross( ray.direction, edge2 );
        double a = dot(edge1, h);

        if ( abs(a) < EPS ) return false;

        double f = 1.0 / a;
        Vec3d s = ray.origin - v1;
        double u = f * dot(s, h);

        if ( u < 0.0 || u > 1.0 ) return false;

        Vec3d q = cross( s, edge1 );
        double v = f * dot(ray.direction, q);

        if  ( v < 0.0 || u + v > 1.0 ) return false;

        double t = f * dot(edge2, q);

        if ( t < EPS ) return false;

        if ( !interval.contains( t ) ) return false;
        record.t = t;
        record.p = ray.at( t );
        record.setFaceNormal( ray, N );

        return true;
    }
#if HIP_ENABLED
    HOST Hittable* copyToDevice() override;

    HOST Hittable* copyToHost() override;

    HOST void deallocateOnDevice() override;
#endif
public:
    void computeBBox();
    void computeNormal();
    Vec3d v1, v2, v3;
    Vec3d N;
};



#endif //COLLECTION_TRIANGLE_H
