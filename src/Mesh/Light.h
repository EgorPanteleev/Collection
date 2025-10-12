#ifndef COLLECTION_LIGHT_H
#define COLLECTION_LIGHT_H
#include "Vector.h"
#include "RGB.h"
#include "Ray.h"

class Light {
public:
    enum Type {
        BASE,
        POINT,
        SPOT
    };

    virtual Type getType() const;

    virtual bool isIntersectsWithRay( const Ray& ray ) const = 0;

    virtual Vec3d getSamplePoint() const = 0;

    double intensity;

    RGB lightColor;
protected:
    Light();
};


#endif //COLLECTION_LIGHT_H
