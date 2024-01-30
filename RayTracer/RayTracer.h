#ifndef COLLECTION_RAYTRACER_H
#define COLLECTION_RAYTRACER_H

#include <utility>

#include "Color.h"
#include "Ray.h"
#include "Scene.h"
#include "Canvas.h"
#include "Camera.h"

struct closestIntersectionData {
    closestIntersectionData():t( std::numeric_limits<float>::max() ), N(), object( nullptr ) {}
    float t;
    Vector3f N;
    Object* object;
};

class RayTracer {
public:
    RayTracer( Camera* c, Scene* s );
    ~RayTracer();
    [[nodiscard]] closestIntersectionData closestIntersection( Ray& ray ) const;
    [[nodiscard]] float computeLight( const Vector3f& P, const Vector3f& V, const closestIntersectionData& iData ) const;
    [[nodiscard]] RGB traceRay( Ray& ray, int depth ) const;
    void traceAllRays();
    [[nodiscard]] Canvas* getCanvas() const;
private:
    Camera* cam;
    Scene* scene;
    Canvas* canvas;
};


#endif //COLLECTION_RAYTRACER_H
