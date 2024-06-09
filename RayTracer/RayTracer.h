#ifndef COLLECTION_RAYTRACER_H
#define COLLECTION_RAYTRACER_H

#include <utility>
#include "Color.h"
#include "Ray.h"
#include "Scene.h"
#include "Canvas.h"
#include "Camera.h"
#include "BVH.h"
#include <Kokkos_Core.hpp>

class RayTracer {
public:
    enum Type {
       SERIAL,
       PARALLEL
    };
    RayTracer( Camera* c, Scene* s, Canvas* _canvas, int _depth, int _numAmbientSamples, int _numLightSamples );
    ~RayTracer();
    [[nodiscard]] IntersectionData closestIntersection( Ray& ray );
    [[nodiscard]] float computeLight( const Vector3f& P, const Vector3f& V, const IntersectionData& iData );
    [[nodiscard]] RGB traceRay( Ray& ray, int nextDepth, float throughput );
    void traceAllRays( Type type );
    [[nodiscard]] Canvas* getCanvas() const;
    [[nodiscard]] Scene* getScene() const;
    [[nodiscard]] Camera* getCamera() const;
private:
    void printProgress( int x ) const;
    void traceAllRaysSerial();
    void traceAllRaysParallel();
    Kokkos::View<Camera*> camera;
    Kokkos::View<Scene*> scene;
    Kokkos::View<Canvas*> canvas;
    Kokkos::View<BVH*> bvh;
    int depth;
    int numAmbientSamples;
    int numLightSamples;
    RGB diffuse;
    RGB ambient;
    RGB specular;
};


#endif //COLLECTION_RAYTRACER_H
