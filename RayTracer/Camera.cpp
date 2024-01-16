#include "Camera.h"
#include "Vector.h"
#include "Mat4f.h"
#include "Utils.h"
Camera::Camera(): viewMatrix(), origin(), forward(), up(), right(), dV(0), Vx(0), Vy(0) {
}
Camera::Camera( Vector3f pos, Vector3f dir, double dv, double vx, double vy ): origin( pos ), forward(dir), dV(dv), Vx(vx), Vy(vy) {
    forward = Vector3f(0,0,1);
    right = Vector3f(1,0,0);
    up = Vector3f(0,1,0);
    LookAt( dir, up);
}

Mat4f Camera::LookAt( Vector3f target, Vector3f _up ) {
    forward = (target - origin).normalize();    // The "forward" vector.
    right = _up.cross(forward).normalize();// The "right" vector.
    up = forward.cross(right);     // The "up" vector.
    // Create a 4x4 view matrix from the right, up, forward and eye position vectors
    viewMatrix = {
            Vector4f(      right[0],            up[0],            forward[0],       0 ),
            Vector4f(      right[1],            up[1],            forward[1],       0 ),
            Vector4f(      right[2],            up[2],            forward[2],       0 ),
            Vector4f(-dot( right, origin ), -dot( up, origin ), -dot( forward, origin ),  1 )
    };
    return viewMatrix;
}

Vector3f Camera::worldToCameraCoordinates( Vector3f& coords ) const {
    Vector3f localCoordinates (viewMatrix[0][0] * coords[0] + viewMatrix[1][0] * coords[1] + viewMatrix[2][0] * coords[2] + viewMatrix[3][0],
                               viewMatrix[0][1] * coords[0] + viewMatrix[1][1] * coords[1] + viewMatrix[2][1] * coords[2] + viewMatrix[3][1],
                               viewMatrix[0][2] * coords[0] + viewMatrix[1][2] * coords[1] + viewMatrix[2][2] * coords[2] + viewMatrix[3][2]);
    return localCoordinates;
}
