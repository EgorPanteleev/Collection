#ifndef COLLECTION_RAYTRACER_H
#define COLLECTION_RAYTRACER_H

#include "Color.h"
#include "Ray.h"
#include "Scene.h"
#include "Canvas.h"
#include "Camera.h"


struct IntersectionData {
    IntersectionData(): t(0), shape(nullptr) {};
    IntersectionData( double _t, Shape* _shape): t(_t), shape(_shape) {};
    double t;
    Shape* shape;
};
class RayTracer {
public:
    RayTracer( Camera* c, Scene* s );
    ~RayTracer();
    IntersectionData closestIntersection( Ray& ray ) const;
    double computeLight( Vector3f P, Vector3f N, Shape* shape );
    RGB traceRay( Ray& ray );
    void traceAllRays();
    Canvas* getCanvas() const;
private:
    Camera* cam;
    Scene* scene;
    Canvas* canvas;
};


#endif //COLLECTION_RAYTRACER_H
