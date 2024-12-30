//
// Created by auser on 11/26/24.
//

#ifndef COLLECTION_CAMERA_H
#define COLLECTION_CAMERA_H
#include "HittableList.h"
#include "Material.h"
#include "SystemUtils.h"
#include "scatter.h"
#include "Mat3.h"
#include "Vec2.h"

class Camera {
public:

    HOST_DEVICE double linearToGamma( double linear ) {
        if ( linear > 0 ) return std::sqrt( linear );
        return 0;
    }

    HOST Camera();

    HOST_DEVICE void init() {
        imageHeight = int( imageWidth / aspectRatio );
        origin = lookFrom;

        double focalLength = ( lookFrom - lookAt ).length();
        auto theta = vFOV * ( M_PI / 180 );
        auto h = std::tan( theta / 2 );
        double viewportHeight = 2.0 * h * focalLength;
        double viewportWidth = viewportHeight * ( double( imageWidth ) / imageHeight );

        forward = ( lookAt - lookFrom ).normalize();
        right = ( cross( globalUp, forward ) ).normalize();
        up = cross( forward, right );

        Vec2<Vec3d> viewport = { viewportWidth * right, viewportHeight * -up };

        pixelDelta[0] = viewport[0] / imageWidth;
        pixelDelta[1] = viewport[1] / imageHeight;
        Vec3d viewportUpperLeft = lookFrom - focalLength * forward  - viewport[0] / 2 - viewport[1] / 2;
        pixelsOrigin = viewportUpperLeft + 0.5 * ( pixelDelta[0] + pixelDelta[1] );
    }

    void move(const Vec3d& direction ) {
        lookFrom += direction[0] * right + direction[1] * up + direction[2] * forward;
        lookAt += direction[0] * right + direction[1] * up + direction[2] * forward;
        init();
    }

    void rotateYaw(double angleRad) {
        auto rotation = Mat3d::rotateY(angleRad);
        forward = rotation * forward;
        lookAt = lookFrom + forward;
        init();
    }

    void rotatePitch(double angleRad) {

        double newPitch = atan2(forward[1], sqrt(forward[0] * forward[0] + forward[2] * forward[2]));
        newPitch += angleRad;

        if (newPitch > M_PI_2) newPitch = M_PI_2 - EPS;
        if (newPitch < -M_PI_2) newPitch = -M_PI_2 + EPS;

        double yaw = atan2(forward[2], forward[0]);
        forward = { cos(newPitch) * cos(yaw),
                    sin(newPitch),
                    cos(newPitch) * sin(yaw) };

        lookAt = lookFrom + forward;
        init();
    }

    void rotateRoll(double angleRad) {
        auto rotation = Mat3d::rotateZ(angleRad);
        up = rotation * up;
        init();
    }

    double aspectRatio;
    int imageWidth;
    int imageHeight;
    int samplesPerPixel;
    int maxDepth;
    double vFOV;

    Point3d lookFrom;
    Point3d lookAt;
    Vec3d globalUp;
public:

//    __device__ vec3 color(const ray& r, hitable **world, curandState *local_rand_state) {
//        ray cur_ray = r;
//        vec3 cur_attenuation = vec3(1.0,1.0,1.0);
//        for(int i = 0; i < 50; i++) {
//            hit_record rec;
//            if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) {
//                ray scattered;
//                vec3 attenuation;
//                if(rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state)) {
//                    cur_attenuation *= attenuation;
//                    cur_ray = scattered;
//                }
//                else {
//                    return vec3(0.0,0.0,0.0);
//                }
//            }
//            else {
//                vec3 unit_direction = unit_vector(cur_ray.direction());
//                float t = 0.5f*(unit_direction.y() + 1.0f);
//                vec3 c = (1.0f-t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0);
//                return cur_attenuation * c;
//            }
//        }
//        return vec3(0.0,0.0,0.0); // exceeded recursion
//    }

    DEVICE RGB traceRay( const Ray& ray, const HittableList& world, int depth, hiprandState& state ) {
        const Interval<double> interval( 0.001, 10000 );
        Ray currentRay = ray;
        RGB currentAttenuation = { 1.0, 1.0, 1.0 };
        for ( int i = 0; i < maxDepth; ++i ) {
            HitRecord rec;
            if (world.hit(currentRay, interval, rec)) {
                Ray scattered;
                RGB attenuation;
                if ( scatter( rec.material, currentRay, rec, attenuation, scattered, state ) ) {
                    currentAttenuation *= attenuation;
                    currentRay = scattered;
                }
                else {
                    return { 0.0, 0.0, 0.0 };
                }
            }
            else {
                Vec3d unitDir = ray.direction.normalize();
                auto a = 0.5 * ( unitDir[1] + 1.0 );
                return currentAttenuation * ( ( 1.0 - a ) * RGB( 1, 1, 1 ) + a * RGB( 0.5, 0.7, 1 ) );
            }
        }
        return { 0.0, 0.0, 0.0 };
    }

    HOST_DEVICE Ray getRay( int i, int j, hiprandState& state ) const {
        Point3d offset = { randomDouble( state ) - 0.5, randomDouble( state ) - 0.5, 0 };
        Point3d pixelSample = pixelsOrigin + ( i + offset[0] ) * pixelDelta[0] + ( j + offset[1] ) * pixelDelta[1];
        return { origin, pixelSample - origin };
    }

    Point3d origin;
    Vec3d pixelsOrigin;
    Vec2<Vec3d> pixelDelta;
    Vec3d forward, up, right;

};


#endif //COLLECTION_CAMERA_H
