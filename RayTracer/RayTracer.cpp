#include "RayTracer.h"
#include <cmath>
#include "Utils.h"
#include "Sheduler.h"
//#define BACKGROUND_COLOR RGB(0, 0, 0)
#define BACKGROUND_COLOR RGB(173, 216, 230)
RayTracer::RayTracer( Camera* c, Scene* s ) {
    cam = c;
    scene = s;
//      canvas = new Canvas(240,150);
//    canvas = new Canvas(960,600);
    canvas = new Canvas(1920,1200);
//    canvas = new Canvas(3200,2000);
}

RayTracer::~RayTracer() {
    delete canvas;
}

closestIntersectionData RayTracer::closestIntersection( Ray& ray ) const {
    closestIntersectionData cIData;
    for ( const auto& object: scene->objects ) {
        IntersectionData iData = object->intersectsWithRay(ray);
        if ( iData.t == std::numeric_limits<float>::max()) continue;
        if ( iData.t <= 0.05 ) continue;
        if ( cIData.t < iData.t ) continue;
        cIData.t = iData.t;
        cIData.N = iData.N;
        cIData.object = object;
    }
    return cIData;
}

float RayTracer::computeLight( const Vector3f& P, const Vector3f& V, const closestIntersectionData& iData ) const {
    float i = 0;
    Vector3f N = iData.N;
    for ( auto light: scene->lights ) {
        Ray ray = Ray( light->origin, P - light->origin );
        closestIntersectionData cIData = closestIntersection( ray );
        if ( cIData.object != iData.object ) continue;
        Vector3f L = ( light->origin - P ).normalize();
        float dNL = dot(N, L );
        if ( dNL > 0 ) i += light->intensity * dNL;
//        if ( iData.object->getDiffuse() == -1 ) continue;
//        Vector3f R = ( N * 2 * dot(N, L) - L ).normalize();
//        float dRV = dot(R, V.normalize());
//        if ( dRV > 0 ) i += light->intensity * pow(dRV, iData.object->getDiffuse());
    }
    if ( i > 1 ) i = 1;
    return i;
}

RGB RayTracer::traceRay( Ray& ray, int depth ) const {
    closestIntersectionData cIData = closestIntersection( ray );
    if ( cIData.t == std::numeric_limits<float>::max() ) return BACKGROUND_COLOR;
    Vector3f P = ray.getOrigin() + ray.getDirection() * cIData.t;
    float i = computeLight( P, ray.getDirection() * (-1), cIData );
    RGB localColor = cIData.object->getColor() * i;
    float r = cIData.object->getReflection();
    if ( depth == 0 || r == 0 ) return localColor;
    Vector3f N = cIData.N.normalize();
    Vector3f reflectedDir = ( ray.getDirection() - N * 2 * dot(N, ray.getDirection() ) );
    Ray reflectedRay( P, reflectedDir );
    RGB reflectedColor = traceRay( reflectedRay, depth - 1 );
    return localColor * (1 - r) + reflectedColor * r;
}

void RayTracer::traceRayUtil( void* self, int x, int y, Ray& ray, int depth  ) {
    RGB color = ((RayTracer*)self)->traceRay( ray, depth );
    ((RayTracer*)self)->mutex.lock();
    ((RayTracer*)self)->canvas->setPixel( x, y, color );
    ((RayTracer*)self)->mutex.unlock();
}


void RayTracer::traceAllRaysWithThreads( int numThreads ) {
    float uX = cam->Vx / canvas->getW();
    float uY = cam->Vy / canvas->getH();
    float uX2 = uX / 2.0f;
    float uY2 = uY / 2.0f;
    Vector3f from = cam->origin;
    Sheduler sheduler(numThreads);
    for ( int x = 0; x < canvas->getW(); ++x ) {
        for ( int y = 0; y < canvas->getH(); ++y ) {
            Vector3f dir = { -cam->Vx / 2 + uX2 + x * uX, -cam->Vy / 2 + uY2 + y * uY, cam->dV  };
            dir =  cam->worldToCameraCoordinates( dir );
            Ray ray( from, dir);
            sheduler.addFunction( traceRayUtil, this, x, y, ray, 15 );
        }
    }
    sheduler.run();
}

void RayTracer::traceAllRays() {
    float uX = cam->Vx / canvas->getW();
    float uY = cam->Vy / canvas->getH();
    float uX2 = uX / 2.0f;
    float uY2 = uY / 2.0f;
    Vector3f from = cam->origin;
    for ( int x = 0; x < canvas->getW(); ++x ) {
        for ( int y = 0; y < canvas->getH(); ++y ) {
            Vector3f dir = { -cam->Vx / 2 + uX2 + x * uX, -cam->Vy / 2 + uY2 + y * uY, cam->dV  };
            Ray ray( from, dir);
            RGB color = traceRay( ray, 7 );
            canvas->setPixel( x, y, color );
        }
    }
}

Canvas* RayTracer::getCanvas() const {
    return canvas;
}