#ifndef COLLECTION_RAY_H
#define COLLECTION_RAY_H
#include "Vec3.h"

class Ray {
public:
    Ray();
    Ray(const Vec3d& from, const Vec3d& dir);
    ~Ray();

    Vec3d origin;
    Vec3d direction;
    Vec3d invDirection;
private:
};


#endif //COLLECTION_RAY_H
