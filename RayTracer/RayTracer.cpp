#include "RayTracer.h"
#include <cmath>
#include "Utils.h"
#define BACKGROUND_COLOR RGB(0,0,0)
#include "Mat.h"
RayTracer::RayTracer( Camera* c, Scene* s ) {
    cam = c;
    scene = s;
    canvas = new Canvas(700,700);
}

RayTracer::~RayTracer() {
    delete canvas;
}

IntersectionData RayTracer::closestIntersection( Ray& ray ) const {
    IntersectionData data( std::numeric_limits<double>::max(), nullptr);
    for ( auto shape: scene->shapes ) {
        double t = shape->intersectsWithRay(ray);
        if (t == std::numeric_limits<float>::min()) continue;
        if ( data.t < t ) continue;
        data.t = t;
        data.shape = shape;
    }
    return data;
}

double RayTracer::computeLight( Vector3f P, Vector3f N, Shape* shape ) {
    double i = 0;
    for ( auto light: scene->lights ) {
        Ray ray = Ray( light->origin, P );
        IntersectionData iData = closestIntersection( ray );
        if ( iData.shape != shape ) continue;
        Vector3f L = light->origin - P;
        double d = dot(N.normalize(), L.normalize() );
        if ( d > 0 ) i += light->intensity * d;
    }
    if ( i > 1 ) i = 1;
    return i;
}

RGB RayTracer::traceRay( Ray& ray ) {
    IntersectionData iData = closestIntersection( ray );
    if ( iData.t == std::numeric_limits<double>::max() ) return BACKGROUND_COLOR;
    Vector3f P = ray.getOrigin() + ray.getDirection() * iData.t;
    Vector3f N = iData.shape->getNormal( P );
    double i = computeLight( P, N, iData.shape );
    return iData.shape->getColor() * i;
}

void RayTracer::traceAllRays() {
    float uX = cam->Vx / canvas->getW();
    float uY = cam->Vy / canvas->getH();
    Vector3f from = cam->origin;
    for ( int x = 0; x < canvas->getW(); ++x ) {
        for ( int y = 0; y < canvas->getH(); ++y ) {
            Vector3f translate = { -cam->Vx / 2 + x * uX, -cam->Vy / 2 + y * uY, cam->dV  };
            translate = cam->worldToCameraCoordinates( translate );
            Vector3f to = from + translate;
            Ray ray( cam->cameraToWorldCoordinates(from), cam->cameraToWorldCoordinates(to));
            RGB color = traceRay( ray );
            canvas->setPixel( x, y, color );
        }
    }
}

Canvas* RayTracer::getCanvas() const {
    return canvas;
}