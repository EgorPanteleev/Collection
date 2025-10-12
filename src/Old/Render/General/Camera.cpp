#include "Camera.h"
#include "Vector.h"
#include "Mat4.h"

#include <stdlib.h>
Camera::Camera(): viewMatrix(), origin(), forward(), up(), right(), dV(0), Vx(0), Vy(0) {
}
Camera::Camera( const Vec3d& pos, const Vec3d& dir, double dv, double vx, double vy ): origin( pos ), forward(dir), dV(dv), Vx(vx), Vy(vy) {
    forward = Vec3d(0,0,1);
    right = Vec3d(1,0,0);
    up = Vec3d(0,1,0);
    if ( dir.normalize() == up )  LookAt( dir, forward * ( -1 ));
    else if ( dir.normalize() == up * ( -1 )) LookAt( dir, forward * ( 1 ) );
    else LookAt( dir, up * ( 1 ) );
}

Mat4f Camera::LookAt( const Vec3d& target, const Vec3d& _up ) {
    forward = (target - origin).normalize();
    right = cross( _up, forward ).normalize();
    up = cross( forward, right );
    viewMatrix = {
            Vec4d(right[0]   ,right[1]   ,right[2]   ,-dot( right  , origin) ),
            Vec4d(up[0]      ,up[1]      ,up[2]      ,-dot( up     , origin) ),
            Vec4d(forward[0] ,forward[1] ,forward[2] ,-dot( forward, origin) ),
            Vec4d(0          ,0          ,0          ,1                            )
    };
    return viewMatrix;
}

Vec3d Camera::worldToCameraCoordinates( Vec3d& coords ) const {
    Vec3d localCoordinates (viewMatrix[0][0] * coords[0] + viewMatrix[1][0] * coords[1] + viewMatrix[2][0] * coords[2] + viewMatrix[3][0],
                               viewMatrix[0][1] * coords[0] + viewMatrix[1][1] * coords[1] + viewMatrix[2][1] * coords[2] + viewMatrix[3][1],
                               viewMatrix[0][2] * coords[0] + viewMatrix[1][2] * coords[1] + viewMatrix[2][2] * coords[2] + viewMatrix[3][2]);
    return localCoordinates;
}

Vec3d Camera::cameraToWorldCoordinates( Vec3d& coords ) const {
    Mat4f vMatrix = viewMatrix.inverse();
    Vec3d globalCoordinates (vMatrix[0][0] * coords[0] + vMatrix[1][0] * coords[1] + vMatrix[2][0] * coords[2] + vMatrix[3][0],
                               vMatrix[0][1] * coords[0] + vMatrix[1][1] * coords[1] + vMatrix[2][1] * coords[2] + vMatrix[3][1],
                               vMatrix[0][2] * coords[0] + vMatrix[1][2] * coords[1] + vMatrix[2][2] * coords[2] + vMatrix[3][2]);
    return globalCoordinates;
}

Vec3d random_in_unit_disk() {
    Vec3d p;
    do {
        p = 2.0 * Vec3d(drand48(), drand48(), 0) - Vec3d(1, 1, 0);
    } while (dot(p, p) >= 1.0);
    return p;
}


Ray Camera::getPrimaryRay( double x, double y ) const {
    Vec3d dir = { -Vx * 0.5f + 0.5f + x, -Vy * 0.5f + 0.5f + y, dV  };
    return { origin, dir };
}

Ray Camera::getSecondaryRay( double x, double y ) const {
    Vec3d rd = aperture * 0.5f  * random_in_unit_disk();
    Vec3d offset = right * rd[0] + up * rd[1];
    Vec3d dir = { -Vx * 0.5f + 0.5f + x, -Vy * 0.5f + 0.5f + y, dV  };
    Vec3d C = origin + focalLenght * dir.normalize();
    return { origin + offset, ( C - origin - offset ).normalize() };
}
