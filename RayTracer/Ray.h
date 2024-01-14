#ifndef COLLECTION_RAY_H
#define COLLECTION_RAY_H
#include "Vector.h"

class Ray {
public:
    Vector3f getOrigin() const;
    Vector3f getDirection() const;
    void setOrigin( Vector3f orig );
    void setDirection( Vector3f dir );
    Ray();
    Ray(Vector3f from, Vector3f to);
    ~Ray();
private:
    Vector3f origin;
    Vector3f direction;
};


#endif //COLLECTION_RAY_H
