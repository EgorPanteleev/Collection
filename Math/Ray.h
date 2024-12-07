#ifndef COLLECTION_RAY_H
#define COLLECTION_RAY_H
#include "Vec3.h"

class Ray {
public:
    HOST_DEVICE Ray():origin(), direction() {}

    HOST_DEVICE Ray(const Vec3d& from, const Vec3d& dir): origin(from), direction( dir.normalize() ) {
    }

    HOST_DEVICE ~Ray() {
        origin = Vec3d();
        direction = Vec3d();
    }

    HOST_DEVICE Vec3d at( double t ) const {
        return origin + t * direction;
    }

    Vec3d origin;
    Vec3d direction;
private:
};


#endif //COLLECTION_RAY_H
