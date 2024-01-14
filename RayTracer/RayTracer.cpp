#include "RayTracer.h"
#include <cmath>
#define BACKGROUND_COLOR RGB(0,0,0)

double dot( Vector3f p1, Vector3f p2 ) {
    return ( p1.getX() * p2.getX() + p1.getY() * p2.getY() + p1.getZ() * p2.getZ());
}

Vector3f normalize( Vector3f p ) {
    double lenght = sqrt( pow( p.getX(), 2 ) +  pow( p.getY(), 2 ) + pow( p.getZ(), 2 ));
    Vector3f p1;
    p1 = p / lenght;
    return p1;
}

RayTracer::RayTracer( Scene* s ) {
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
        double d = dot(normalize( N ), normalize( L ) );
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

void RayTracer::traceAllRays( Camera& cam ) {
    double uX = cam.Vx / canvas->getW();
    double uY = cam.Vy / canvas->getH();
    for ( int x = 0; x < canvas->getW(); ++x ) {
        for ( int y = 0; y < canvas->getH(); ++y ) {
            Vector3f from = cam.origin;
            Vector3f to = Vector3f( from.getX() - cam.Vx / 2 + x * uX,
                              from.getY() - cam.Vy / 2 + y * uY ,
                              from.getZ() + cam.dV );
            Ray ray( from, to);
            RGB color = traceRay( ray );
            canvas->setPixel( x, y, color );
        }
    }
}

Canvas* RayTracer::getCanvas() const {
    return canvas;
}