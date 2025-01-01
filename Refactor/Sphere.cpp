//
// Created by auser on 11/26/24.
//

#include "Sphere.h"

HOST Sphere::Sphere(): Hittable(SPHERE) {}

HOST Sphere::Sphere( double r, const Point3d& pos, Material* mat ): Hittable(SPHERE, mat), radius( r ), origin(pos) {}

#if HIP_ENABLED
HOST Hittable* Sphere::copyToDevice() {
    auto deviceMaterial = material->copyToDevice();
    auto originalMaterial = material;
    material = deviceMaterial;

    auto deviceSphere = HIP::allocateOnDevice<Sphere>();

    HIP::copyToDevice( this, deviceSphere );

    material = originalMaterial;
    return deviceSphere;

}

HOST Hittable* Sphere::copyToHost() {
    auto host = new Sphere();
    HIP::copyToHost( host, this );

    auto hostMaterial = material->copyToHost();
    host->material = hostMaterial;
    return host;
}

HOST void Sphere::deallocateOnDevice() {
    material->deallocateOnDevice();
    HIP::deallocateOnDevice( this );
}
#endif