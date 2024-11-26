#ifndef COLLECTION_RAY_H
#define COLLECTION_RAY_H
#include "Vec3.h"

class Ray {
public:
    Ray();
    Ray(const Vec3d& from, const Vec3d& dir);
    ~Ray();

    Vec3d at( double t ) const;

    Vec3d origin;
    Vec3d direction;
private:
};


#endif //COLLECTION_RAY_H
