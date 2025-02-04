//
// Created by auser on 11/26/24.
//

#ifndef COLLECTION_CAMERA_H
#define COLLECTION_CAMERA_H
#include "BVH.h"
#include "Material.h"
#include "SystemUtils.h"
#include "scatter.h"
#include "Mat3.h"
#include "Vec2.h"


inline DEVICE Vec3d randomInUnitDisk( hiprandState& state ) {
    while (true) {
        randomDouble( -1, 1, state );
        auto p = Vec3d( randomDouble( -1, 1, state ), randomDouble( -1, 1, state ) , 0);
        if ( p.lengthSquared() < 1 ) return p;
    }
}

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

        auto theta = toRadians( vFOV );
        auto h = std::tan( theta / 2 );
        double viewportHeight = 2.0 * h * focusDistance;
        double viewportWidth = viewportHeight * ( double( imageWidth ) / imageHeight );

        forward = ( lookFrom - lookAt ).normalize();
        right = ( cross( globalUp, forward ) ).normalize();
        up = cross( forward, right );

        Vec2<Vec3d> viewport = { viewportWidth * right, viewportHeight * -up };

        pixelDelta[0] = viewport[0] / imageWidth;
        pixelDelta[1] = viewport[1] / imageHeight;
        Vec3d viewportUpperLeft = lookFrom - focusDistance * forward  - viewport[0] / 2 - viewport[1] / 2;
        pixelsOrigin = viewportUpperLeft + 0.5 * ( pixelDelta[0] + pixelDelta[1] );

        auto defocusRadius = focusDistance * std::tan( toRadians( defocusAngle / 2 ) );
        defocusDisk[0] = right * defocusRadius;
        defocusDisk[1] = up * defocusRadius;
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
    double defocusAngle;
    double focusDistance;

    Point3d lookFrom;
    Point3d lookAt;
    Vec3d globalUp;

    RGB background;
public:

//    color ray_color(const ray& r, int depth, const hittable& world) const {
//        // If we've exceeded the ray bounce limit, no more light is gathered.
//        if (depth <= 0)
//            return color(0,0,0);
//
//        hit_record rec;
//
//        // If the ray hits nothing, return the background color.
//        if (!world.hit(r, interval(0.001, infinity), rec))
//            return background;
//
//        ray scattered;
//        color attenuation;
//        color color_from_emission = rec.mat->emitted(rec.u, rec.v, rec.p);
//
//        if (!rec.mat->scatter(r, rec, attenuation, scattered))
//            return color_from_emission;
//
//        color color_from_scatter = attenuation * ray_color(scattered, depth-1, world);
//
//        return color_from_emission + color_from_scatter;
//    }


    DEVICE RGB traceRay( const Ray& ray, const BVH& world, hiprandState& state ) {
        const Interval<double> interval( 0.001, INF );
        Ray currentRay = ray;
        RGB currentAttenuation = { 1, 1, 1 };
        for ( int i = 0; i < maxDepth; ++i ) {
            HitRecord rec;
            if ( world.hit(currentRay, interval, rec ) ) {
                Ray scattered;
                RGB attenuation;
                if ( scatter( rec.material, currentRay, rec, attenuation, scattered, state ) ) {
                    currentAttenuation *= attenuation;
                    currentRay = scattered;
                } else {
                    RGB emission = emit( rec.material, rec.u, rec.v, rec.p );
                    return currentAttenuation * emission;
                }
            }
            else {
                return currentAttenuation * background;
            }
        }
        return 0;
    }

    HOST_DEVICE Ray getRay( int i, int j, hiprandState& state ) const {
        Point3d offset = { randomDouble( state ) - 0.5, randomDouble( state ) - 0.5, 0 };
        Point3d pixelSample = pixelsOrigin + ( i + offset[0] ) * pixelDelta[0] + ( j + offset[1] ) * pixelDelta[1];
        Point3d rayOrigin = ( defocusAngle <= 0) ? lookFrom : defocusDiskSample( state );
        return { rayOrigin, pixelSample - rayOrigin };
    }

    DEVICE Point3d defocusDiskSample( hiprandState& state ) const {
        auto p = randomInUnitDisk( state );
        return lookFrom + ( p[0] * defocusDisk[0] ) + ( p[1] * defocusDisk[1] );
    }

    Point3d origin;
    Vec3d pixelsOrigin;
    Vec2<Vec3d> pixelDelta;
    Vec2<Vec3d> defocusDisk;
    Vec3d forward, up, right;

};


#endif //COLLECTION_CAMERA_H
