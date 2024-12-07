//
// Created by auser on 11/26/24.
//

#include "Camera.h"

//HOST_DEVICE double linearToGamma( double linear ) {
//    if ( linear > 0 ) return std::sqrt( linear );
//    return 0;
//}

//Vec3d randomOnHemisphere( const Vec3d& N ) {
//    Vec3d onUnitSphere = randomUnitVector();
//    if ( dot( onUnitSphere, N ) > 0 )
//        return onUnitSphere;
//    return -onUnitSphere;
//}


//HOST_DEVICE void Camera::writeColor( unsigned char* colorBuffer, const RGB& color, int i, int j, int imageWidth ) {
//    int index = (j * imageWidth + i) * 4;
//    static const Interval<double> intensity( 0, 0.999 );
//    colorBuffer[index + 0] = (unsigned char) ( intensity.clamp( linearToGamma( color.r ) ) * 256 );
//    colorBuffer[index + 1] = (unsigned char) ( intensity.clamp( linearToGamma( color.g ) ) * 256 );
//    colorBuffer[index + 2] = (unsigned char) ( intensity.clamp( linearToGamma( color.b ) ) * 256 );
//    colorBuffer[index + 3] = 255;
//}

HOST Camera::Camera() {
}


DEVICE void Camera::render( const HittableList& world, unsigned char* colorBuffer ) {
    const double pixelSamplesScale = 1.0 / samplesPerPixel;
    for ( int i = 0; i < imageWidth; ++i ) {
        for ( int j = 0; j < imageHeight; ++j ) {
            RGB pixelColor = { 0, 0, 0 };
            for ( int s = 0; s < samplesPerPixel; ++s ) {
                Vec3d pixelCenter = pixel00Loc + (i * pixelDeltaU) + (j * pixelDeltaV);
                //Ray ray = getRay( i, j );
               // RGB color = traceRay(ray, world, maxDepth, 0, 0 );
                //std::cout << color.r << " " << color.b << " " << color.g << std::endl;
                //pixelColor += color;
            }
            writeColor(colorBuffer, pixelColor * pixelSamplesScale, i, j, imageWidth);
        }
    }
}

//HOST_DEVICE RGB Camera::traceRay( const Ray& ray, const HittableList& world, int depth ) {
//    Interval<double> interval( 0.001, std::numeric_limits<double>::infinity() );
//    HitRecord hitRecord;
//    if ( depth <= 0 ) return { 0 ,0, 0 };
//    if ( world.hit( ray, interval, hitRecord ) ) {
//        Ray scattered;
//        RGB attenuation;
//        if ( hitRecord.material->scatter( ray, hitRecord, attenuation, scattered ) ) {
//            return attenuation * traceRay( scattered, world, depth - 1 );
//        }
//        return { 0, 0, 0 };
//    }
//
//
//    Vec3d unitDir = ray.direction.normalize();
//    auto a = 0.5 * ( unitDir[1] + 1.0 );
//    return ( 1.0 - a ) * RGB( 1, 1, 1 ) + a * RGB( 0.5, 0.7, 1 );
//}

//HOST_DEVICE Ray Camera::getRay( int i, int j ) const {
//    Point3d offset = { randomDouble() - 0.5, randomDouble() - 0.5, 0 };
//    Point3d pixelSample = pixel00Loc + ( i + offset[0] ) * pixelDeltaU + ( j + offset[1] ) * pixelDeltaV;
//    return { origin, pixelSample - origin };
//}

//HOST_DEVICE void Camera::init() {
//    imageHeight = int( imageWidth / aspectRatio );
//    origin = lookFrom;
//
//    double focalLength = ( lookFrom - lookAt ).length();
//    auto theta = vFOV * ( M_PI / 180 );
//    auto h = std::tan( theta / 2 );
//    double viewportHeight = 2.0 * h * focalLength;
//    double viewportWidth = viewportHeight * ( double( imageWidth ) / imageHeight );
//
//    w = ( lookFrom - lookAt ).normalize();
//    u = ( cross( up, w ) ).normalize();
//    v = cross( w, u );
//
//    Vec3d viewportU = viewportWidth * u;
//    Vec3d viewportV = viewportHeight * -v;
//    pixelDeltaU = viewportU / imageWidth;
//    pixelDeltaV = viewportV / imageHeight;
//    Vec3d viewportUpperLeft = lookFrom - focalLength * w  - viewportU / 2 - viewportV / 2;
//    pixel00Loc = viewportUpperLeft + 0.5 * ( pixelDeltaU + pixelDeltaV );
//}
