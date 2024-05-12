#ifndef COLLECTION_RAY_H
#define COLLECTION_RAY_H
#include "Vector.h"

class Ray {
public:
    Ray();
    Ray(const Vector3f& from, const Vector3f& dir);
    ~Ray();

    Vector3f origin;
    Vector3f direction;
private:
};


#endif //COLLECTION_RAY_H
