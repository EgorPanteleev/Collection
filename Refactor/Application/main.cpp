//
// Created by auser on 11/26/24.
//

#include <iostream>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "Sphere.h"
#include "Timer.h"
#include "Camera.h"

//RGB traceRay( const Ray& ray, const HittableList& world ) {
//    Interval<double> interval( 0, std::numeric_limits<double>::infinity() );
//    HitRecord hitRecord;
//    if ( world.hit( ray, interval, hitRecord ) ) {
//        Vec3d tmp = 0.5 * ( hitRecord.N + Vec3d( 1, 1, 1 ) );
//        return { tmp[0], tmp[1], tmp[2] };
//    }
//
//
//    Vec3d unitDir = ray.direction.normalize();
//    auto a = 0.5 * ( unitDir[1] + 1.0 );
//    return ( 1.0 - a ) * RGB( 1, 1, 1 ) + a * RGB( 0.5, 0.7, 1 );
//}
//
//void writeColor( unsigned char* colorBuffer, const RGB& color, int i, int j, int imageWidth ) {
//    int index = (j * imageWidth + i) * 4;
//    colorBuffer[index + 0] = (unsigned char) ( color.r * 255 );
//    colorBuffer[index + 1] = (unsigned char) ( color.g * 255 );
//    colorBuffer[index + 2] = (unsigned char) ( color.b * 255 );
//    colorBuffer[index + 3] = 255;
//}

void saveToPNG( const std::string& fileName, unsigned char* colorBuffer, int imageWidth, int imageHeight ) {
    if (stbi_write_png( fileName.c_str(), imageWidth, imageHeight, 4, colorBuffer, imageWidth * 4))
        std::cout << "Image saved successfully: " << fileName << std::endl;
    else std::cerr << "Failed to save image: " << fileName << std::endl;
}

int main() {
    srand(time( nullptr ));
//    //init list
    HittableList world;
    Lambertian* ground = new Lambertian( { 0.8, 0.8, 0.0 } );
    Lambertian* center = new Lambertian( { 0.1, 0.2, 0.5 } );
    Dielectric* left = new Dielectric( 1.5 );
    Dielectric* bubble = new Dielectric( 1.0 / 1.5 );
    Metal* right = new Metal( { 0.8, 0.6, 0.2 }, 1.0 );
    world.add( new Sphere( 100, { 0, -100.5, -1 }, ground ) );
    world.add( new Sphere( 0.5, { 0, 0, -1.2 }, center ) );
    world.add( new Sphere( 0.5, { -1, 0, -1 }, left ) );
    world.add( new Sphere( 0.4, { -1, 0, -1 }, bubble ) );
    world.add( new Sphere( 0.5, { 1, 0, -1 }, right ) );
    //

    //camera settings
    Camera cam;
    cam.aspectRatio = 16.0 / 10.0;
    cam.imageWidth = 800;
    cam.samplesPerPixel = 100;
    cam.maxDepth = 30;
    cam.vFOV = 30;

    cam.lookFrom = { -2, 2, 1 };
    cam.lookAt = { 0, 0, -1 };
    cam.up = { 0, 1, 0 };

    cam.init();

    auto colorBuffer = new unsigned char[ cam.imageWidth * cam.imageHeight * 4 ];

    Timer timer;

    timer.start();

    //cam.render( world, colorBuffer );

    timer.end();

    std::cout << "RayTracer works "<< timer.get() << " seconds" << std::endl;

    saveToPNG( "out.png", colorBuffer, cam.imageWidth, cam.imageHeight );

}