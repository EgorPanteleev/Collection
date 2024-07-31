#ifndef COLLECTION_LIGHT_H
#define COLLECTION_LIGHT_H
#include "Vector.h"
#include "Color.h"
#include "Ray.h"

class Light {
public:
    enum Type {
        BASE,
        POINT,
        SPOT
    };

    __host__ __device__ virtual Type getType() const;

    __host__ __device__ virtual bool isIntersectsWithRay( const Ray& ray ) const = 0;

    __device__ virtual Vector3f getSamplePoint() const = 0;

    float intensity;

    RGB lightColor;
protected:
    __host__ __device__ Light();
};


#endif //COLLECTION_LIGHT_H
