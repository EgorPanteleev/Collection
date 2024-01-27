#ifndef COLLECTION_RAYTRACER_H
#define COLLECTION_RAYTRACER_H

#include "Color.h"
#include "Ray.h"
#include "Scene.h"
#include "Canvas.h"
#include "Camera.h"


struct IntersectionData {
    IntersectionData(): t(0), shape(nullptr) {};
    IntersectionData( float _t, Shape* _shape): t(_t), shape(_shape) {};
    float t;
    Shape* shape;
};
class RayTracer {
public:
    RayTracer( Camera* c, Scene* s );
    ~RayTracer();
    [[nodiscard]] IntersectionData closestIntersection( Ray& ray ) const;
    [[nodiscard]] float computeLight( const Vector3f& P, const Vector3f& N, Shape* shape ) const;
    [[nodiscard]] RGB traceRay( Ray& ray ) const;
    void traceAllRays();
    [[nodiscard]] Canvas* getCanvas() const;
private:
    Camera* cam;
    Scene* scene;
    Canvas* canvas;
};


#endif //COLLECTION_RAYTRACER_H
