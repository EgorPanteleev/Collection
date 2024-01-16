#include "RayTracer.h"
#include <cmath>
#include "Utils.h"
#define BACKGROUND_COLOR RGB(0,0,0)
#include "Mat.h"
RayTracer::RayTracer( Camera* c, Scene* s ) {
    cam = c;
    scene = s;
    canvas = new Canvas(2000,2000);
}

RayTracer::~RayTracer() {
    delete canvas;
}

double RayTracer::computeLight( Vector3f P, Vector3f N ) {
    double i = 0;
    for ( auto light: scene->lights ) {
        Vector3f L = light->origin - P;
        double d = dot(N.normalize(), L.normalize() );
        if ( d > 0 ) i += light->intensity * d;
    }
    if ( i > 1 ) i = 1;
    return i;
}

RGB RayTracer::traceRay( Ray& ray ) {
    for ( auto shape: scene->shapes ) {
        double t = shape->intersectsWithRay( ray );
        if ( t != std::numeric_limits<double>::min() ) {
            Vector3f P = ray.getOrigin() + ray.getDirection() * t;
            Vector3f N = shape->getNormal( P );
            double i = computeLight( P, N );
            return shape->getColor() * i;
        }
    }
    return BACKGROUND_COLOR;
}

void RayTracer::traceAllRays() {
    float uX = cam->Vx / canvas->getW();
    float uY = cam->Vy / canvas->getH();
    for ( int x = 0; x < canvas->getW(); ++x ) {
        for ( int y = 0; y < canvas->getH(); ++y ) {
            Vector3f from = cam->origin;
            Vector3f translate = { -cam->Vx / 2 + x * uX, -cam->Vy / 2 + y * uY, cam->dV  };
            translate = cam->worldToCameraCoordinates( translate );
            Vector3f to = from + translate;

            Ray ray( from, to);
            RGB color = traceRay( ray );
            canvas->setPixel( x, y, color );
        }
    }
}

Canvas* RayTracer::getCanvas() const {
    return canvas;
}