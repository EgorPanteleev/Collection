//
// Created by auser on 11/26/24.
//

#ifndef COLLECTION_CAMERA_H
#define COLLECTION_CAMERA_H
#include "HittableList.h"
#include "RGB.h"

class Camera {
public:
    Camera();
    void render( const HittableList& world, unsigned char* colorBuffer );

    void init();

    double aspectRatio;
    int imageWidth;
    int imageHeight;
    int samplesPerPixel;
    int maxDepth;
    double vFOV;

    Point3d lookFrom;
    Point3d lookAt;
    Vec3d up;
private:
    RGB traceRay( const Ray& ray, const HittableList& world, int depth );
    Ray getRay( int i, int j ) const;

    Point3d origin;
    Vec3d pixel00Loc;
    Vec3d pixelDeltaU;
    Vec3d pixelDeltaV;

    Vec3d u, v, w;

};


#endif //COLLECTION_CAMERA_H
