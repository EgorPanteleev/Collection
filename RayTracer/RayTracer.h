#ifndef COLLECTION_RAYTRACER_H
#define COLLECTION_RAYTRACER_H

#include <utility>

#include "Color.h"
#include "Ray.h"
#include "Scene.h"
#include "Canvas.h"
#include "Camera.h"
#include <mutex>
#include "BVH.h"

class RayTracer {
public:
    RayTracer( Camera* c, Scene* s, Canvas* _canvas );
    ~RayTracer();
    [[nodiscard]] IntersectionData closestIntersection( Ray& ray );
    [[nodiscard]] float computeLight( const Vector3f& P, const Vector3f& V, const IntersectionData& iData );
    [[nodiscard]] RGB traceRay( Ray& ray, int depth );
    static void traceRayUtil( void* self, int x, int y, Ray& ray, int depth );
    void traceAllRaysWithThreads( int numThreads );
    void traceAllRays();
    [[nodiscard]] Canvas* getCanvas() const;
    [[nodiscard]] Scene* getScene() const;
    [[nodiscard]] Camera* getCamera() const;
private:
    std::mutex mutex;
    Camera* camera;
    Scene* scene;
    Canvas* canvas;
    BVH* bvh;
};


#endif //COLLECTION_RAYTRACER_H
