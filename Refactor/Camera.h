//
// Created by auser on 11/26/24.
//

#ifndef COLLECTION_CAMERA_H
#define COLLECTION_CAMERA_H
#include "HittableList.h"
#include "Material.h"
#include "SystemUtils.h"
#include "scatter.h"
#include "Mat4.h"

class Camera {
public:

    HOST_DEVICE double linearToGamma( double linear ) {
        if ( linear > 0 ) return std::sqrt( linear );
        return 0;
    }

    HOST_DEVICE void writeColor( unsigned char* colorBuffer, const RGB& color, int i, int j, int imageWidth ) {
        int index = (j * imageWidth + i) * 4;
        const Interval<double> intensity( 0, 0.999 );
        colorBuffer[index + 0] = (unsigned char) ( intensity.clamp( linearToGamma( color[0] ) ) * 256 );
        colorBuffer[index + 1] = (unsigned char) ( intensity.clamp( linearToGamma( color[1] ) ) * 256 );
        colorBuffer[index + 2] = (unsigned char) ( intensity.clamp( linearToGamma( color[2] ) ) * 256 );
        colorBuffer[index + 3] = 255;
    }

    HOST Camera();
    DEVICE void render( const HittableList& world, unsigned char* colorBuffer );

    HOST_DEVICE void init() {
        imageHeight = int( imageWidth / aspectRatio );
        origin = lookFrom;

        double focalLength = ( lookFrom - lookAt ).length();
        auto theta = vFOV * ( M_PI / 180 );
        auto h = std::tan( theta / 2 );
        double viewportHeight = 2.0 * h * focalLength;
        double viewportWidth = viewportHeight * ( double( imageWidth ) / imageHeight );

        w = ( lookFrom - lookAt ).normalize();
        u = ( cross( up, w ) ).normalize();
        v = cross( w, u );

        Vec3d viewportU = viewportWidth * u;
        Vec3d viewportV = viewportHeight * -v;
        pixelDeltaU = viewportU / imageWidth;
        pixelDeltaV = viewportV / imageHeight;
        Vec3d viewportUpperLeft = lookFrom - focalLength * w  - viewportU / 2 - viewportV / 2;
        pixel00Loc = viewportUpperLeft + 0.5 * ( pixelDeltaU + pixelDeltaV );
    }

    void move(const Vec3d& direction ) {
        lookFrom += direction[0] * u + direction[1] * v + direction[2] * w;
        lookAt += direction[0] * u + direction[1] * v + direction[2] * w;
        init();
    }

    void rotateYaw(double angle) {
        Mat4d rotation = Mat4d::rotateY(angle);
        auto forward =   Vec4d( ( lookAt - lookFrom ).normalize() );
        forward = rotation * forward;
        forward = forward.normalize();
        lookAt = { lookFrom[0] + forward[0], lookFrom[1] + forward[1], lookFrom[2] + forward[2] };
        init();
    }

    void rotatePitch(double angle) {
        Mat4d rotation = Mat4d::rotateX(angle);
        auto forward = Vec4d( ( lookAt - lookFrom ).normalize() );
        forward = rotation * forward;
        auto up4 = Vec4d( up );
        up4 = rotation * up4;
        forward = forward.normalize();
        forward = forward.normalize();
        up = { up4[0], up4[1], up4[2] };
        up = up.normalize();
        lookAt = { lookFrom[0] + forward[0], lookFrom[1] + forward[1], lookFrom[2] + forward[2] };
        init();
    }

    void rotateRoll(double angle) {
        Mat4d rotation = Mat4d::rotateZ(angle);
        auto up4 = Vec4d( up );
        up4 = rotation * up4;
        up = { up4[0], up4[1], up4[2] };
        up = up.normalize();
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
    Vec3d up;
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
        Interval<double> interval( 0.001, std::numeric_limits<double>::infinity() );
        Ray currentRay = ray;
        RGB currentAttenuation = { 1.0, 1.0, 1.0 };
        for(int i = 0; i < maxDepth; i++) {
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
        return { 0.0, 0.0, 0.0 }; // exceeded recursion
    }

//    HOST_DEVICE RGB traceRay( const Ray& ray, const HittableList& world, int depth ) {
//        Interval<double> interval( 0.001, std::numeric_limits<double>::infinity() );
//        HitRecord hitRecord;
//        if ( depth <= 0 ) return { 0, 0, 0 };
//        if ( world.hit( ray, interval, hitRecord ) ) {
//            Ray scattered;
//            RGB attenuation;
//            if ( hitRecord.material->scatter( ray, hitRecord, attenuation, scattered ) ) {
//                return attenuation * traceRay( scattered, world, depth - 1 );
//            }
//            return { 1, 0, 0 };
//        }
//
//        Vec3d unitDir = ray.direction.normalize();
//        auto a = 0.5 * ( unitDir[1] + 1.0 );
//        return ( 1.0 - a ) * RGB( 1, 1, 1 ) + a * RGB( 0.5, 0.7, 1 );
//    }

    HOST_DEVICE Ray getRay( int i, int j, hiprandState& state ) const {
        Point3d offset = { randomDouble( state ) - 0.5, randomDouble( state ) - 0.5, 0 };
        Point3d pixelSample = pixel00Loc + ( i + offset[0] ) * pixelDeltaU + ( j + offset[1] ) * pixelDeltaV;
        return { origin, pixelSample - origin };
    }

//    HOST_DEVICE static void writeColor( unsigned char* colorBuffer, const RGB& color, int i, int j, int imageWidth ) {
//        int index = (j * imageWidth + i) * 4;
//        static const Interval<double> intensity( 0, 0.999 );
//        colorBuffer[index + 0] = (unsigned char) ( intensity.clamp( linearToGamma( color.r ) ) * 256 );
//        colorBuffer[index + 1] = (unsigned char) ( intensity.clamp( linearToGamma( color.g ) ) * 256 );
//        colorBuffer[index + 2] = (unsigned char) ( intensity.clamp( linearToGamma( color.b ) ) * 256 );
//        colorBuffer[index + 3] = 255;
//    }

    Point3d origin;
    Vec3d pixel00Loc;
    Vec3d pixelDeltaU;
    Vec3d pixelDeltaV;

    Vec3d u, v, w;

};


#endif //COLLECTION_CAMERA_H
