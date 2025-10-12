//
// Created by auser on 1/5/25.
//

#include "Triangle.h"


Triangle::Triangle(): Hittable(TRIANGLE), v1(), v2(), v3(), N() {

}

Triangle::Triangle( const Vec3d& v1, const Vec3d& v2, const Vec3d& v3 ): Hittable(TRIANGLE), v1( v1 ), v2( v2 ), v3( v3 ) {
    computeNormal();
    computeBBox();
}

Triangle::Triangle( const Vec3d& v1, const Vec3d& v2, const Vec3d& v3, Material* mat ): Hittable(TRIANGLE, mat), v1( v1 ), v2( v2 ), v3( v3 ) {
    computeNormal();
    computeBBox();
}


[[nodiscard]] Vec3d Triangle::getNormal() const {
    return N;
}

void Triangle::computeBBox() {
    bbox.merge( v1 );
    bbox.merge( v2 );
    bbox.merge( v3 );
}

void Triangle::computeNormal() {
    N = cross( v2 - v1, v3 - v1 ).normalize();
}

#if HIP_ENABLED
HOST Hittable* Triangle::copyToDevice() {
    auto deviceMaterial = material->copyToDevice();
    auto originalMaterial = material;
    material = deviceMaterial;

    auto device = HIP::allocateOnDevice<Triangle>();

    HIP::copyToDevice( this, device );

    material = originalMaterial;
    return device;

}

HOST Hittable* Triangle::copyToHost() {
    auto host = new Triangle();
    HIP::copyToHost( host, this );

    auto hostMaterial = material->copyToHost();
    host->material = hostMaterial;
    return host;
}

HOST void Triangle::deallocateOnDevice() {
    material->deallocateOnDevice();
    HIP::deallocateOnDevice( this );
}
#endif
