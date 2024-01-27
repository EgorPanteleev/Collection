#ifndef COLLECTION_CAMERA_H
#define COLLECTION_CAMERA_H
#include "Vector.h"
#include "Mat4f.h"
class Camera {
public:
    Camera();
    Camera( Vector3f pos, Vector3f dir, double dv, double vx, double vy );
    Mat4f LookAt( Vector3f target, Vector3f up );
    Vector3f worldToCameraCoordinates( Vector3f& coords ) const;
    Vector3f cameraToWorldCoordinates( Vector3f& coords ) const;
public:
    Mat4f viewMatrix;
    Vector3f origin;
    Vector3f forward;
    Vector3f up;
    Vector3f right;
    float dV;
    float Vx;
    float Vy;
};


#endif //COLLECTION_CAMERA_H
