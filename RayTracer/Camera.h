#ifndef COLLECTION_CAMERA_H
#define COLLECTION_CAMERA_H
#include "Vector.h"

class Camera {
public:
    Camera();
    Camera( Vector3f pos, double dv, double vx, double vy );
public:
    Vector3f origin;
    double dV;
    double Vx;
    double Vy;
};


#endif //COLLECTION_CAMERA_H
