#ifndef COLLECTION_RAYTRACER_H
#define COLLECTION_RAYTRACER_H

#include <utility>

#include "Color.h"
#include "Ray.h"
#include "Scene.h"
#include "Canvas.h"
#include "Camera.h"


struct IntersectionData {
    IntersectionData(): t(0), object() {};
    IntersectionData( float _t, Object* _object): t(_t), object(_object) {};
    float t;
    Object* object;
};

class RayTracer {
public:
    RayTracer( Camera* c, Scene* s );
    ~RayTracer();
    [[nodiscard]] IntersectionData closestIntersection( Ray& ray ) const;
    [[nodiscard]] float computeLight( const Vector3f& P, const Vector3f& V, Object* object ) const;
    [[nodiscard]] RGB traceRay( Ray& ray, int depth ) const;
    void traceAllRays();
    [[nodiscard]] Canvas* getCanvas() const;
private:
    Camera* cam;
    Scene* scene;
    Canvas* canvas;
};


#endif //COLLECTION_RAYTRACER_H
