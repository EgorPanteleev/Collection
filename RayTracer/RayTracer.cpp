#include "RayTracer.h"

#define BACKGROUND_COLOR RGB(0,0,0)

RayTracer::RayTracer( Scene* s ) {
    scene = s;
    canvas = new Canvas(500,500);
}

RayTracer::~RayTracer() {
    delete canvas;
}
RGB RayTracer::traceRay( Ray& ray ) {
    for ( auto shape: scene->shapes ) {
        if ( shape->intersectsWithRay( ray ) != std::numeric_limits<double>::min() ) {
            return shape->getColor();
        }
    }
    return BACKGROUND_COLOR;
}

void RayTracer::traceAllRays( Camera& cam ) {
    double uX = cam.Vx / canvas->getW();
    double uY = cam.Vy / canvas->getH();
    for ( int x = 0; x < canvas->getW(); ++x ) {
        for ( int y = 0; y < canvas->getH(); ++y ) {
            Point from = cam.origin;
            Point to = Point( from.getX() - cam.Vx / 2 + x * uX,
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