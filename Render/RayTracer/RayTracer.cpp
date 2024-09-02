#include "RayTracer.h"
#include <cmath>
#include "Utils.h"
#include "LuaLoader.h"
#include "Sampler.h"
extern "C" {
#include "lua.h"
#include "lualib.h"
#include "lauxlib.h"
}

//#include "Sheduler.h"
#define BACKGROUND_COLOR RGB(0, 0, 0)
//#define BACKGROUND_COLOR RGB(255, 0, 0)
//#define BACKGROUND_COLOR RGB(255, 255, 255)
//TODO no return, передавать - не возвращать
RayTracer::RayTracer( Camera* c, Scene* s, Canvas* _canvas, int _depth, int _numAmbientSamples, int _numLightSamples ) {
    load( c, s, _canvas, _depth, _numAmbientSamples, _numLightSamples );
}
RayTracer::RayTracer( const std::string& path ) {
    lua_State* L = luaL_newstate();
    luaL_openlibs(L);

    if (luaL_dofile(L, path.c_str() ) != LUA_OK) {
        std::cerr << lua_tostring(L, -1) << std::endl;
        lua_close(L);
        return;
    }
    Scene* s = new Scene();
    loadScene( L, s );
    Canvas* can = loadCanvas( L );
    Camera* cam = loadCamera( L, can->getW(), can->getH() );
    auto settings = loadSettings( L );
    lua_close(L);
    load( cam, s, can, settings[0], settings[1], settings[2] );

}
RayTracer::~RayTracer() {
    //delete canvas;
}

void RayTracer::closestIntersection( Ray& ray, IntersectionData& iData ) {
    bvh->intersectBVH( ray, iData, 0 );
}

template <typename Type>
float getIntensity( Type* light ) {
    return light->getIntensity();
}


RGB RayTracer::computeDiffuseLight( const Vector3f& V, const TraceData& traceData ) {
    RGB i;
    float constexpr del = 1.0f / 255;
    float roughness = traceData.getRoughness();
    float metalness = traceData.getMetalness();
    IntersectionData cIData;
    for ( auto light: scene(0).getLights() ) {
        RGB i1;
        for ( size_t j = 0; j < numLightSamples; j++ ) {
            Vector3f origin = light->getSamplePoint();
            Vector3f L = (origin - traceData.P).normalize();
            Ray ray = Ray(origin, L * ( -1 ) );
            closestIntersection( ray, cIData );
            if ( !cIData.primitive ) continue;
            //remove
            Vector3f P1 = ray.origin + cIData.t * ray.direction;
            if ( getDistance( traceData.P, P1 ) > 1e-3 ) continue;
            //
            float distance = cIData.t * 0.01f;
            float inverseDistance2 = 1.0f / ( distance * distance );
            Vector2f alpha = { pow2( roughness ), pow2( roughness ) };
            Vector3f wo_T = traceData.cs.to( V );
            Vector3f wi_T = traceData.cs.to( L );
            float Nwi = saturate(wi_T.z);
            if ( Nwi < 0 ) continue;
            if ( roughness == 1 && metalness == 0 )  {
                i1 = i1 + Lambertian::BRDF() * light->getIntensity() * ( light->getColor() * del ) * Nwi * inverseDistance2;
            }
            if ( roughness != 1 && metalness < 0.5 ) {
                i1 = i1 + ( 1 - metalness ) * OrenNayar::BRDF( wi_T, wo_T, alpha[0] ) * light->getIntensity() * ( light->getColor() * del ) * Nwi * inverseDistance2;
            }
            if ( roughness != 1 && metalness >= 0.5 ) {
                float PDF;
                i1 = i1 + metalness * GGX::BRDF( wi_T, wo_T, alpha, PDF ) * light->getIntensity() * ( light->getColor() * del ) * Nwi * inverseDistance2;
            }
        }
        i = i + i1 / (float) numLightSamples;
    }
    //if ( i > 1 ) i = 1;
    return i;
}

RGB RayTracer::computeAmbientLight( const Ray& ray, const TraceData& traceData, int nextDepth ) {
    RGB ambient = {};
    float roughness = traceData.getRoughness();
    float metalness = traceData.getMetalness();
    if ( roughness == 1 ) ambient = ambient + computeDiffuseLambertian( ray, traceData, nextDepth );
    if ( roughness != 1 && metalness != 1 ) ambient = ambient + ( 1 - metalness ) * computeDiffuseOrenNayar( ray, traceData, nextDepth );
    if ( roughness != 1 && metalness != 0 ) ambient = ambient + ( metalness ) * computeReflectanceGGX( ray, traceData, nextDepth );
    return ambient;
}

KOKKOS_INLINE_FUNCTION RGB RayTracer::computeReflectanceGGX( const Ray& ray, const TraceData& traceData, int nextDepth ) {
    RGB ambient = {};
    Vector3f wo_T = traceData.cs.to( ray.direction * (-1) );
    Vector2f alpha = { pow2( traceData.getRoughness() ), pow2( traceData.getRoughness() ) };
    for (int j = 0; j < numAmbientSamples; j++ ) {
        Vector3f H_T = GGX::getNormal( wo_T, alpha );
        Vector3f wi_T = reflect( wo_T, H_T ) * ( -1 );
        float Nwi = saturate(wi_T.z);
        float PDF;
        float BRDF = GGX::BRDF( wi_T, wo_T, alpha, PDF );
        Vector3f wi = traceData.cs.from( wi_T );
        Ray newRay = { traceData.P + wi * 1e-3, wi };
        CanvasData data;
        traceRay( newRay, data, nextDepth - 1 );
        ambient = ambient + BRDF / PDF * data.color * Nwi;
    }
    return ambient * traceData.ambientOcclusion / (float) numAmbientSamples;
}
KOKKOS_INLINE_FUNCTION RGB RayTracer::computeDiffuseOrenNayar( const Ray& ray, const TraceData& traceData, int nextDepth ) {
    RGB ambient = {};
    Vector3f wo_T = traceData.cs.to( ray.direction * (-1) );
    for (int j = 0; j < numAmbientSamples; j++ ) {
        Vector3f wi_T = OrenNayar::getIncidentDir( traceData.cs.getNormal() );
        Vector3f wi = traceData.cs.from( wi_T );
        float Nwi = saturate(wi_T.z);
        float PDF = OrenNayar::PDF( Nwi );
        float BRDF = OrenNayar::BRDF( wi_T, wo_T, traceData.getRoughness() );
        Ray newRay = { traceData.P + wi * 1e-3, wi };
        CanvasData data;
        traceRay( newRay, data, nextDepth - 1 );
        ambient = ambient + BRDF / PDF * data.color * Nwi;
    }
    return ambient * traceData.ambientOcclusion / (float) numAmbientSamples;
}

KOKKOS_INLINE_FUNCTION RGB RayTracer::computeDiffuseLambertian( const Ray& ray, const TraceData& traceData, int nextDepth ) {
    RGB ambient = {};
    for (int j = 0; j < numAmbientSamples; j++ ) {
        Vector3f wi_T = Lambertian::getIncidentDir( traceData.cs.getNormal() );
        Vector3f wi = traceData.cs.from( wi_T );
        float Nwi = saturate(wi_T.z);
        float PDF = Lambertian::PDF( Nwi );
        float BRDF = Lambertian::BRDF();
        Ray newRay = { traceData.P + wi * 1e-3, wi };
        CanvasData data;
        traceRay( newRay, data, nextDepth - 1 );
        ambient = ambient + BRDF / PDF * data.color * Nwi;
    }
    return ambient * traceData.ambientOcclusion / (float) numAmbientSamples;
}
//TODO нужна реорганизация для оптимизации
void RayTracer::traceRay( Ray& ray, CanvasData& data, int nextDepth ) {
    IntersectionData iData;
    closestIntersection( ray, iData );
    if ( !iData.primitive ) {
        data = { BACKGROUND_COLOR, { 0, 0, 0 }, BACKGROUND_COLOR };
        return;
    }
    TraceData tData;
    tData.t = iData.t;
    tData.P = ray.origin + tData.t * ray.direction;
    tData.primitive = iData.primitive;
    tData.load();
    Vector3f vectorColor = ( tData.cs.getNormal() + Vector3f( 1, 1, 1 ) ) * 255 * 0.5f;
    RGB normalColor = { vectorColor.x, vectorColor.y, vectorColor.z };
    if ( tData.material.getIntensity() != 0 ) {
        data = { tData.getColor(), normalColor, tData.getColor() };
        return;
    }
    RGB i = computeDiffuseLight( ray.direction * (-1), tData );
    RGB diffuse = {};
    diffuse = tData.material.getColor() * i;
    if ( nextDepth == 0 ) {
        data = { diffuse, normalColor, tData.getColor() };
        return;
    }
    //Global illumination
    RGB ambient = computeAmbientLight( ray, tData, nextDepth );
    data = { diffuse + tData.getColor() / 255 * ambient, normalColor, tData.getColor() };
}

void RayTracer::load( Camera* c, Scene* s, Canvas* _canvas, int _depth, int _numAmbientSamples, int _numLightSamples ) {
    camera = Kokkos::View<Camera*>("camera");
    Kokkos::deep_copy(camera, *c);
    scene = Kokkos::View<Scene*>("scene");
    Kokkos::deep_copy(scene, *s);
    canvas = Kokkos::View<Canvas*>("canvas");
    Kokkos::deep_copy(canvas, *_canvas);
    bvh = new BVH( s->getPrimitives() );
    depth = _depth;
    numAmbientSamples = _numAmbientSamples;
    numLightSamples = _numLightSamples;
}

void RayTracer::printProgress( int x ) const {
    std::cout << "Progress: " << ( (float) ( x + 1 ) / canvas(0).getW() ) * 100 << std::endl;
}

void RayTracer::render( Type type ) {
    switch (type) {
        case Type::SERIAL: {
            traceAllRaysSerial();
            break;
        }
        case Type::PARALLEL: {
            traceAllRaysParallel();
            break;
        }
    }
}

void RayTracer::traceAllRaysSerial() {
    for ( int x = 0; x < canvas(0).getW(); ++x ) {
        printProgress( x );
        for ( int y = 0; y < canvas(0).getH(); ++y ) {
            Ray ray = getCamera()->getPrimaryRay( x, y );
            CanvasData colorData;
            traceRay( ray, colorData, depth );
            colorData.color.scaleTo( 255 );
            canvas(0).setColor( x, y, colorData.color );
            canvas(0).setNormal( x, y, colorData.normal );
            canvas(0).setAlbedo( x, y, colorData.albedo );
        }
    }
}
void RayTracer::traceAllRaysParallel() {
    Kokkos::View<RGB**> colors = Kokkos::View<RGB**>("colors", canvas(0).getW(), canvas(0).getH() );
    Kokkos::View<RGB**> normals = Kokkos::View<RGB**>("normals", canvas(0).getW(), canvas(0).getH() );
    Kokkos::View<RGB**> albedos = Kokkos::View<RGB**>("albedos", canvas(0).getW(), canvas(0).getH() );
    RenderFunctor renderFunctor( this, colors, normals, albedos );
    typedef Kokkos::MDRangePolicy<Kokkos::Rank<2>> range_policy_2d;
    range_policy_2d policy({0, 0}, {canvas(0).getW(), canvas(0).getH()});
    Kokkos::parallel_for("parallel2D", policy, renderFunctor);
    for ( int i = 0; i < canvas(0).getW(); i++ ) {
        for ( int j = 0; j < canvas(0).getH(); j++ ) {
            canvas(0).setColor( i, j, colors(i, j) );
            canvas(0).setNormal( i, j, normals(i, j) );
            canvas(0).setAlbedo( i, j, albedos(i, j) );
        }
    }
}

Canvas* RayTracer::getCanvas() const {
    return &(canvas(0));
}

Scene* RayTracer::getScene() const {
    return &(scene(0));
}
Camera* RayTracer::getCamera() const {
    return &(camera(0));
}

int RayTracer::getDepth() const {
    return depth;
}


RenderFunctor::RenderFunctor( RayTracer* _rayTracer, Kokkos::View<RGB**>& _colors, Kokkos::View<RGB**>& _normals, Kokkos::View<RGB**>& _albedos )
        :rayTracer( _rayTracer ), colors( _colors ), normals( _normals ), albedos( _albedos ) {
}

void RenderFunctor::operator()(const int i, const int j) const {
//    int numSamples = 3;
//    for ( int sample = 0; sample < numSamples; sample++ ) {
//        Ray ray = rayTracer->getCamera()->getPrimaryRay(i, j);
////        if ( sample == 0 ) {
////            ray = rayTracer->getCamera()->getPrimaryRay(i, j);
////        } else {
////            ray = rayTracer->getCamera()->getSecondaryRay(i, j);
////        }
//        CanvasData data = rayTracer->traceRay(ray, rayTracer->getDepth() );
//        colors(i, j) = colors(i, j) +  data.color / (float) numSamples;
//        normals(i, j) = normals(i, j) +  data.normal / (float) numSamples;
//        albedos(i, j) = albedos(i, j) +  data.albedo / (float) numSamples;
//    }
    //printf("array(%d, %d) = %f\n", i, j, colors(i,j).r);
    Ray ray = rayTracer->getCamera()->getPrimaryRay(i, j);
    CanvasData data;
    rayTracer->traceRay(ray, data, rayTracer->getDepth() );
    colors(i, j) = data.color;
    normals(i, j) = data.normal;
    albedos(i, j) = data.albedo;
}
