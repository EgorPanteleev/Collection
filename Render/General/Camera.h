#ifndef COLLECTION_CAMERA_H
#define COLLECTION_CAMERA_H
#include "Vector3f.h"
#include "Mat4f.h"
class Camera {
public:
    __host__ __device__ Camera();
    __host__ __device__ Camera( const Vector3f& pos, const Vector3f& dir, float dv, float vx, float vy );
    __host__ __device__ Mat4f LookAt( const Vector3f& target, const Vector3f& up );
    [[nodiscard]] __host__ __device__ Vector3f worldToCameraCoordinates( Vector3f& coords ) const;
    [[nodiscard]] __host__ __device__ Vector3f cameraToWorldCoordinates( Vector3f& coords ) const;
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
