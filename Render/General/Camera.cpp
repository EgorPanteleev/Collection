#include "Camera.h"
#include "Vector.h"
#include "Mat4f.h"
#include "Utils.h"
__host__ __device__ Camera::Camera(): viewMatrix(), origin(), forward(), up(), right(), dV(0), Vx(0), Vy(0) {
}
__host__ __device__ Camera::Camera( const Vector3f& pos, const Vector3f& dir, float dv, float vx, float vy ): origin( pos ), forward(dir), dV(dv), Vx(vx), Vy(vy) {
    forward = Vector3f(0,0,1);
    right = Vector3f(1,0,0);
    up = Vector3f(0,1,0);
    if ( dir.normalize() == up )  LookAt( dir, forward * ( -1 ));
    else if ( dir.normalize() == up * ( -1 )) LookAt( dir, forward * ( 1 ) );
    else LookAt( dir, up * ( 1 ) );
}

__host__ __device__ Mat4f Camera::LookAt( const Vector3f& target, const Vector3f& _up ) {
    forward = (target - origin).normalize();
    right = _up.cross(forward).normalize();
    up = forward.cross(right);
    viewMatrix = {
            Vector4f(right[0]   ,right[1]   ,right[2]   ,-dot( right  , origin) ),
            Vector4f(up[0]      ,up[1]      ,up[2]      ,-dot( up     , origin) ),
            Vector4f(forward[0] ,forward[1] ,forward[2] ,-dot( forward, origin) ),
            Vector4f(0          ,0          ,0          ,1                            )
    };
    return viewMatrix;
}

__host__ __device__ Vector3f Camera::worldToCameraCoordinates( Vector3f& coords ) const {
    Vector3f localCoordinates (viewMatrix[0][0] * coords[0] + viewMatrix[1][0] * coords[1] + viewMatrix[2][0] * coords[2] + viewMatrix[3][0],
                               viewMatrix[0][1] * coords[0] + viewMatrix[1][1] * coords[1] + viewMatrix[2][1] * coords[2] + viewMatrix[3][1],
                               viewMatrix[0][2] * coords[0] + viewMatrix[1][2] * coords[1] + viewMatrix[2][2] * coords[2] + viewMatrix[3][2]);
    return localCoordinates;
}

__host__ __device__ Vector3f Camera::cameraToWorldCoordinates( Vector3f& coords ) const {
    Mat4f vMatrix = viewMatrix.inverse();
    Vector3f globalCoordinates (vMatrix[0][0] * coords[0] + vMatrix[1][0] * coords[1] + vMatrix[2][0] * coords[2] + vMatrix[3][0],
                               vMatrix[0][1] * coords[0] + vMatrix[1][1] * coords[1] + vMatrix[2][1] * coords[2] + vMatrix[3][1],
                               vMatrix[0][2] * coords[0] + vMatrix[1][2] * coords[1] + vMatrix[2][2] * coords[2] + vMatrix[3][2]);
    return globalCoordinates;
}
