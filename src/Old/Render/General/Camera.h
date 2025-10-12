#ifndef COLLECTION_CAMERA_H
#define COLLECTION_CAMERA_H
#include "Vec3.h"
#include "Mat4.h"
#include "Ray.h"
class Camera {
public:
    Camera();
    Camera( const Vec3d& pos, const Vec3d& dir, double dv, double vx, double vy );
    Mat4f LookAt( const Vec3d& target, const Vec3d& up );
    Ray getPrimaryRay( double x, double y ) const;
    Ray getSecondaryRay( double x, double y ) const;
    [[nodiscard]] Vec3d worldToCameraCoordinates( Vec3d& coords ) const;
    [[nodiscard]] Vec3d cameraToWorldCoordinates( Vec3d& coords ) const;
public:
    Mat4f viewMatrix;
    Vec3d origin;
    Vec3d forward;
    Vec3d up;
    Vec3d right;
    double dV;
    double Vx;
    double Vy;
    double focalLenght = 225;
    double aperture = 0;
};


#endif //COLLECTION_CAMERA_H
