#ifndef COLLECTION_LIGHT_H
#define COLLECTION_LIGHT_H
#include "Vector.h"

class Light {
public:
    Light();
    Light( const Vector3f& origin, float intensity );
    Vector3f origin;
    float intensity;
private:
};


#endif //COLLECTION_LIGHT_H
