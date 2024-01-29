#include "RayTracer.h"
#include <cmath>
#include "Utils.h"
//#define BACKGROUND_COLOR RGB(0, 0, 0)
#define BACKGROUND_COLOR RGB(173, 216, 230)
RayTracer::RayTracer( Camera* c, Scene* s ) {
    cam = c;
    scene = s;
    canvas = new Canvas(2000,2000);
}

RayTracer::~RayTracer() {
    delete canvas;
}

IntersectionData RayTracer::closestIntersection( Ray& ray ) const {
    IntersectionData data( std::numeric_limits<float>::max(), nullptr);
    for ( const auto& object: scene->objects ) {
        float t = object->intersectsWithRay(ray);
        if (t == std::numeric_limits<float>::min()) continue;
        if ( t < 0 ) continue;
        if ( data.t < t ) continue;
        data.t = t;
        data.object = object;
    }
    return data;
}

float RayTracer::computeLight( const Vector3f& P, const Vector3f& V, Object* object ) const {
    float i = 0;
    Vector3f N = object->getNormal( P ).normalize();
    for ( auto light: scene->lights ) {
        Ray ray = Ray( light->origin, P - light->origin );
        IntersectionData iData = closestIntersection( ray );
        if ( iData.object != object ) continue;
        Vector3f L = ( light->origin - P ).normalize();
        float dNL = dot(N, L );
        if ( dNL > 0 ) i += light->intensity * dNL;
        if ( object->getDiffuse() == -1 ) continue;
        Vector3f R = ( N * 2 * dot(N, L) - L ).normalize();
        float dRV = dot(R, V.normalize());
        if ( dRV > 0 ) i += light->intensity * pow(dRV, object->getDiffuse());
    }
    if ( i > 1 ) i = 1;
    return i;
}

RGB RayTracer::traceRay( Ray& ray, int depth ) const {
    IntersectionData iData = closestIntersection( ray );
    if ( iData.t == std::numeric_limits<float>::max() ) return BACKGROUND_COLOR;
    Vector3f P = ray.getOrigin() + ray.getDirection() * iData.t;
    float i = computeLight( P, ray.getDirection() * (-1), iData.object );
    RGB localColor = iData.object->getColor() * i;
    float r = iData.object->getReflection();
    if ( depth == 0 || r == 0 ) return localColor;
    Vector3f N = iData.object->getNormal( P ).normalize();
    Vector3f reflectedDir = ( ray.getDirection() - N * 2 * dot(N, ray.getDirection() ) );
    Ray reflectedRay( P, reflectedDir );
    RGB reflectedColor = traceRay( reflectedRay, depth - 1 ) * 0.7;
    return localColor * (1 - r) + reflectedColor * r;
}

void RayTracer::traceAllRays() {
    float uX = cam->Vx / canvas->getW();
    float uY = cam->Vy / canvas->getH();
    Vector3f from = cam->origin;
    for ( int x = 0; x < canvas->getW(); ++x ) {
        for ( int y = 0; y < canvas->getH(); ++y ) {
            Vector3f dir = { -cam->Vx / 2 + x * uX, -cam->Vy / 2 + y * uY, cam->dV  };
            Ray ray( from, dir);
            RGB color = traceRay( ray, 5 );
            canvas->setPixel( x, y, color );
        }
    }
}

Canvas* RayTracer::getCanvas() const {
    return canvas;
}