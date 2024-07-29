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
    KOKKOS_INLINE_FUNCTION IntersectionData closestIntersection( Ray& ray );
    KOKKOS_INLINE_FUNCTION float computeLight( const Vector3f& P, const Vector3f& V, const IntersectionData& iData );
    KOKKOS_INLINE_FUNCTION RGB traceRay( Ray& ray, int nextDepth, float throughput );
    void render( Type type );
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
    BVH* bvh;
    int depth;
    int numAmbientSamples;
    int numLightSamples;
    RGB diffuse;
    RGB ambient;
    RGB specular;
};


struct RenderFunctor {

    RenderFunctor(float _uX, float _uY, float _uX2, float _uY2, float _Vx2,
                  float _Vy2, float _dV, int _depth, Vector3f _from, RayTracer* _rayTracer, Kokkos::View<RGB**>& result );


    KOKKOS_INLINE_FUNCTION void operator()(const int i, const int j) const;

    Kokkos::View<RGB**> colors;
    RayTracer* rayTracer;
    Vector3f from;
    int depth;
    float uX, uY, uX2, uY2, Vx2, Vy2, dV;
};


#endif //COLLECTION_RAYTRACER_H
