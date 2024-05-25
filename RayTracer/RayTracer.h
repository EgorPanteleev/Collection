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
    RayTracer( Camera* c, Scene* s, Canvas* _canvas, int _depth, int _numAmbientSamples, int _numLightSamples );
    ~RayTracer();
    [[nodiscard]] IntersectionData closestIntersection( Ray& ray );
    [[nodiscard]] float computeLight( const Vector3f& P, const Vector3f& V, const IntersectionData& iData );
    [[nodiscard]] RGB traceRay( Ray& ray, int nextDepth, float throughput );
    static void traceRayUtil( void* self, int x, int y, Ray& ray, int nextDepth );
    void traceAllRaysWithThreads( int numThreads );
    void traceAllRays();
    [[nodiscard]] Canvas* getCanvas() const;
    [[nodiscard]] Scene* getScene() const;
    [[nodiscard]] Camera* getCamera() const;
private:
    void printProgress( int x ) const;
    std::mutex mutex;
    Camera* camera;
    Scene* scene;
    Canvas* canvas;
    BVH* bvh;
    int depth;
    int numAmbientSamples;
    int numLightSamples;
    RGB diffuse;
    RGB ambient;
    RGB specular;
};


#endif //COLLECTION_RAYTRACER_H
