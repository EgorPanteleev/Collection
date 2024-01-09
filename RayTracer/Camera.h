#ifndef COLLECTION_CAMERA_H
#define COLLECTION_CAMERA_H
#include "Point.h"

class Camera {
public:
    Camera();
    Camera( Point pos, double dv, double vx, double vy );
public:
    Point origin;
    double dV;
    double Vx;
    double Vy;
};


#endif //COLLECTION_CAMERA_H
