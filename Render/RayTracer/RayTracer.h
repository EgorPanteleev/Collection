#ifndef COLLECTION_RAYTRACER_H
#define COLLECTION_RAYTRACER_H

#include <utility>
#include "Color.h"
#include "CoordinateSystem.h"
#include "Ray.h"
#include "Scene.h"
#include "Canvas.h"
#include "Camera.h"
#include "BVH.h"
#include <Kokkos_Core.hpp>

struct TraceData {
    TraceData(): cs(), material(), intersection(), ambientOcclusion() {}
    TraceData( const Triangle* tri, const Vector3f& P ) {
        intersection = P;
        cs = { tri->getNormal( P ) };
        material.setColor( tri->getColor( P ) );
        material.setIntensity( tri->getMaterial().getIntensity() );
        material.setRoughness( tri->getRoughness( P ) );
        material.setMetalness( tri->getMetalness( P ) );
        ambientOcclusion = tri->getAmbient( P ).r;
    }
    TraceData( const Sphere* sph, const Vector3f& P ) {
        intersection = P;
        cs = { sph->getNormal( P ) };
        material.setColor( sph->getColor( P ) );
        material.setIntensity( sph->getMaterial().getIntensity() );
        material.setRoughness( sph->getRoughness( P ) );
        material.setMetalness( sph->getMetalness( P ) );
        ambientOcclusion = sph->getAmbient( P ).r;
    }
    [[nodiscard]] RGB getColor() const {
        return material.getColor();
    }
    [[nodiscard]] float getIntensity() const {
        return material.getIntensity();
    }
    [[nodiscard]] float getRoughness() const {
        return material.getRoughness();
    }
    [[nodiscard]] float getMetalness() const {
        return material.getMetalness();
    }
    CoordinateSystem cs;
    Material material;
    Vector3f intersection;
    float ambientOcclusion;
};

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
    KOKKOS_INLINE_FUNCTION RGB computeDiffuseLight( const Vector3f& V, const TraceData& traceData );
    KOKKOS_INLINE_FUNCTION RGB computeAmbientLight( const Ray& ray, const TraceData& traceData, int nextDepth );
    KOKKOS_INLINE_FUNCTION CanvasData traceRay( Ray& ray, int nextDepth );
    void render( Type type );
    [[nodiscard]] Canvas* getCanvas() const;
    [[nodiscard]] Scene* getScene() const;
    [[nodiscard]] Camera* getCamera() const;
    [[nodiscard]] int getDepth() const;
private:
    KOKKOS_INLINE_FUNCTION RGB computeReflectanceGGX( const Ray& ray, const TraceData& traceData, int nextDepth );
    KOKKOS_INLINE_FUNCTION RGB computeDiffuseOrenNayar( const Ray& ray, const TraceData& traceData, int nextDepth );
    KOKKOS_INLINE_FUNCTION RGB computeDiffuseLambertian( const Ray& ray, const TraceData& traceData, int nextDepth );
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
