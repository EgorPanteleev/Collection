#ifndef COLLECTION_RAY_H
#define COLLECTION_RAY_H
#include "Vector3f.h"

class Ray {
public:
    __host__ __device__ Ray();
    __host__ __device__ Ray(const Vector3f& from, const Vector3f& dir);
    __host__ __device__ ~Ray();

    Vector3f origin;
    Vector3f direction;
private:
};


#endif //COLLECTION_RAY_H
