#ifndef COLLECTION_RAY_H
#define COLLECTION_RAY_H
#include "Vector.h"

class Ray {
public:
    [[nodiscard]] Vector3f getOrigin() const;
    [[nodiscard]] Vector3f getDirection() const;
    void setOrigin( const Vector3f& orig );
    void setDirection( const Vector3f& dir );
    Ray();
    Ray(const Vector3f& from, const Vector3f& dir);
    ~Ray();
private:
    Vector3f origin;
    Vector3f direction;
};


#endif //COLLECTION_RAY_H
