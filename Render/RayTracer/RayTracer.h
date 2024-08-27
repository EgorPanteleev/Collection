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

struct CanvasData {
    RGB color;
    RGB normal;
    RGB albedo;
};

class RayTracer {
public:
    enum Type {
       SERIAL,
       PARALLEL
    };
    RayTracer( Camera* c, Scene* s, Canvas* _canvas, int _depth, int _numAmbientSamples, int _numLightSamples );
    RayTracer( const std::string& path  );
    ~RayTracer();
    KOKKOS_INLINE_FUNCTION IntersectionData closestIntersection( Ray& ray );
    KOKKOS_INLINE_FUNCTION RGB computeDiffuseLight( const Vector3f& P, const Vector3f& V, const IntersectionData& iData );
    KOKKOS_INLINE_FUNCTION RGB computeAmbientLight( const Ray& ray, const IntersectionData& iData, float roughness, float ambientOcclusion, int nextDepth );
    KOKKOS_INLINE_FUNCTION CanvasData traceRay( Ray& ray, int nextDepth );
    void render( Type type );
    [[nodiscard]] Canvas* getCanvas() const;
    [[nodiscard]] Scene* getScene() const;
    [[nodiscard]] Camera* getCamera() const;
    [[nodiscard]] int getDepth() const;
private:
    KOKKOS_INLINE_FUNCTION RGB computeReflectanceGGX( const Ray& ray, const IntersectionData& iData, float roughness, float ambientOcclusion, int nextDepth );
    KOKKOS_INLINE_FUNCTION RGB computeDiffuseOrenNayar( const Ray& ray, const IntersectionData& iData, float roughness, float ambientOcclusion, int nextDepth );
    KOKKOS_INLINE_FUNCTION RGB computeDiffuseLambertian( const Ray& ray, const IntersectionData& iData, float roughness, float ambientOcclusion, int nextDepth );
    void load( Camera* c, Scene* s, Canvas* _canvas, int _depth, int _numAmbientSamples, int _numLightSamples );
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
    //RGB diffuse;
    //RGB ambient;
    //RGB specular;
};


struct RenderFunctor {

    RenderFunctor( RayTracer* _rayTracer, Kokkos::View<RGB**>& _colors, Kokkos::View<RGB**>& _normals, Kokkos::View<RGB**>& _albedos );


    KOKKOS_INLINE_FUNCTION void operator()(const int i, const int j) const;

    Kokkos::View<RGB**> colors;
    Kokkos::View<RGB**> normals;
    Kokkos::View<RGB**> albedos;
    RayTracer* rayTracer;
};


#endif //COLLECTION_RAYTRACER_H
