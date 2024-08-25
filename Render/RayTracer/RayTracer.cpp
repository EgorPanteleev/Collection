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

RayTracer::RayTracer( Camera* c, Scene* s, Canvas* _canvas, int _depth, int _numAmbientSamples, int _numLightSamples  ) {
    load( c, s, _canvas, _depth, _numAmbientSamples, _numLightSamples );
}
RayTracer::RayTracer( const std::string& path  ) {
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

IntersectionData RayTracer::closestIntersection( Ray& ray ) {
    return bvh->intersectBVH( ray, 0 );
}

template <typename Type>
float getIntensity( Type* light ) {
    return light->getIntensity();
}

//TODO READ THIS
//есть диффузная составляющая, есть specular, думаю это все зависит от roughness, но почитать, я не уверен
// все остальное считается как ambient light ( specular - reflection, diffuse - diffuse )
// nado разбить все по пальчикам так сказать, реализовать диффузное рассеивание при помощи cosineweighted( было реализовано)
//добавить specular рассеивание и НОРМ БУДЕТ)
RGB RayTracer::computeDiffuseLight( const Vector3f& P, const Vector3f& V, const IntersectionData& iData ) {
    RGB i;
    Vector3f N = iData.N;
    float constexpr del = 1.0f / 255;
    float roughness;
    if ( iData.triangle != nullptr ) {
        roughness = iData.triangle->getRoughness( P );
    } else {
        roughness = iData.sphere->getRoughness( P );
    }
    for ( auto light: scene(0).getLights() ) {
        RGB i1;
        for ( size_t j = 0; j < numLightSamples; j++ ) {
            Vector3f origin = light->getSamplePoint();
            Vector3f L = (origin - P).normalize();
            float dNL = dot( iData.N, L);
            if ( dNL < 0 ) continue;
            Ray ray = Ray(origin, L * ( -1 ) );
            IntersectionData cIData = closestIntersection( ray );
            if ( cIData.t == __FLT_MAX__ ) continue;
            if (cIData.triangle != iData.triangle ) continue;
            if (cIData.sphere != iData.sphere ) continue;
            float distance = cIData.t * 0.01f;
            float inverseDistance2 = 1 / ( distance * distance );

            //lambert


            i1 = i1 + Lambertian::BRDF() * light->getIntensity() * ( light->getColor() * del ) * dNL * inverseDistance2;
            //i1 = i1 + OrenNayar::BRDF( iData.N, L, V, roughness * roughness ) * light->getIntensity() * ( light->getColor() * del ) * dNL * inverseDistance2;

            //specular
            //if (iData.triangle->owner->getMaterial().getDiffuse() == -1) continue;
//
//            Vector3f H = (L + V ).normalize();
//            //float power = 1 / roughness;
//            //float power = 2.0f / pow( roughness, 4 ) - 2; //better
//            auto a2 = (float) pow( roughness, 4 );
//            float dNH = dot( N, H );
//
//            auto X = []( float val ) {
//                return ( val > 0 ) ? 1 : 0;
//            };
//
//
//            //float D = 1.0f / ( M_PI * a2 ) * pow( dNH, power ); //BLINN
//
//            float tan = ( dNH * dNH - 1 ) / ( dNH * dNH );
//            //float D = 1.0f / ( M_PI * a2 ) * exp( tan / a2 ) / pow( dNH, 4 ); //BECKMANN
//
//            float D = ( a2 * (float) X( dNH ) ) / (float) ( M_PI * pow( ( dNH * dNH * ( a2 - tan ) ), 2 ) ); //GGX
//
//            float dNV = dot( N, V );
//
//            float dVH = dot( V, H ); // really H, ost M
//
//            //float G = std::min( 1.0f, std::min(  2 * dNH * dNV / dVH, 2 * dNH * dNL / dVH  ) );  //cook-torrance
//
//            auto G1 = [=]( const Vector3f& x ) {
//                float dNX = dot(x,N);
//                return X(dot(x,H)/dNX) * 2 / ( 1 + sqrt( 1 - a2 * ( dNX * dNX - 1 ) / ( dNX * dNX ) ) );
//            };
//
//            float G = G1( V ) * G1( L ); //GGX
//            float refraction = 0.5;
//            float F0 = pow( refraction - 1, 2 ) / pow( refraction + 1, 2 );
//            float F = F0 + ( 1 - F0 ) * pow( 1 - dVH, 5 );
//
//            float rs = ( D * G * F ) / ( 4 * dNL * dNV );
//            //TODO idk about dist
//            i1 = i1 + s * rs * light->getIntensity() * ( light->getColor() * del ) * dNL * inverseDistance2;
//            Vector3f R = (N * 2 * dNL - L).normalize();
//            float dRV = dot(R, V.normalize());
//            if (dRV > 0) i1 += light->intensity * pow(dRV, iData.triangle->owner->getMaterial().getDiffuse());
        }
        i = i + i1 / (float) numLightSamples;
    }
    //if ( i > 1 ) i = 1;
    return i;
}

float saturate(float z) {
    if (z < 0.0f) return 0.0f;
    if (z > 1.0f) return 1.0f;
    return z;
}


RGB RayTracer::computeAmbientLight( const Ray& ray, const IntersectionData& iData, float roughness, float ambientOcclusion, float throughput, int nextDepth ) {
    RGB ambient = {};
    if ( roughness < 0.6 ) ambient = ambient + computeReflectanceGGX( ray, iData, roughness, ambientOcclusion, throughput, nextDepth );
    if ( roughness >= 0.6 && roughness != 1 ) ambient = ambient + computeDiffuseOrenNayar( ray, iData, roughness, ambientOcclusion, throughput, nextDepth );
    if ( roughness == 1 ) ambient = ambient + computeDiffuseLambertian( ray, iData, roughness, ambientOcclusion, throughput, nextDepth );
    return ambient;
}


KOKKOS_INLINE_FUNCTION RGB RayTracer::computeReflectanceGGX( const Ray& ray, const IntersectionData& iData, float roughness, float ambientOcclusion, float throughput, int nextDepth ) {
    RGB ambient = {};
    Vector3f P = ray.origin + iData.t * ray.direction;
    Vector3f N = iData.N;
    Vector3f wo = ray.direction;
    Mat3f TBN = getTBN( wo, N );
    Vector3f wo_T = (-1) * wo * TBN;
    Vector2f alpha = { roughness * roughness, roughness * roughness};
    for (int j = 0; j < numAmbientSamples; j++ ) {
        if ( roughness == 1 ) continue;
        alpha = { roughness * roughness, roughness * roughness};
        Vector3f H_T = GGX::getNormal( wo_T, alpha );
        Vector3f wi_T = reflect( wo_T, H_T ) * ( -1 );
        float Nwi = saturate(wi_T.z);
        if ( Nwi < 0 ) continue;
        float PDF;
        float BRDF = GGX::BRDF( wi_T, wo_T, alpha, PDF );
        Vector3f wi = TBN * wi_T;
        Ray newRay = { P + wi * 1e-3, wi };
        ambient = ambient + BRDF / PDF * traceRay( newRay, nextDepth - 1, throughput * 0.8f ).color * Nwi;
    }
    return ambient * ambientOcclusion / (float) numAmbientSamples;
}
KOKKOS_INLINE_FUNCTION RGB RayTracer::computeDiffuseOrenNayar( const Ray& ray, const IntersectionData& iData, float roughness, float ambientOcclusion, float throughput, int nextDepth ) {
    RGB ambient = {};
    Vector3f P = ray.origin + iData.t * ray.direction;
    Vector3f N = iData.N;
    Vector3f wo = ray.direction * (-1);
    for (int j = 0; j < numAmbientSamples; j++ ) {
        Vector3f wi = OrenNayar::getIncidentDir( N );
        float Nwi = dot( wi, N );
        if ( Nwi < 0 ) continue;
        float PDF = OrenNayar::PDF( Nwi );
        float BRDF = OrenNayar::BRDF( N, wi, wo, roughness );
        Ray newRay = { P + wi * 1e-3, wi };
        ambient = ambient + BRDF / PDF * traceRay( newRay, nextDepth - 1, throughput * 0.8f ).color * Nwi;
    }
    return ambient * ambientOcclusion / (float) numAmbientSamples;
}

KOKKOS_INLINE_FUNCTION RGB RayTracer::computeDiffuseLambertian( const Ray& ray, const IntersectionData& iData, float roughness, float ambientOcclusion, float throughput, int nextDepth ) {
    RGB ambient = {};
    Vector3f P = ray.origin + iData.t * ray.direction;
    Vector3f N = iData.N;
    Vector3f wo = ray.direction * (-1);
    for (int j = 0; j < numAmbientSamples; j++ ) {
        Vector3f wi = Lambertian::getIncidentDir( N );
        float Nwi = dot( wi, N );
        if ( Nwi < 0 ) continue;
        float PDF = Lambertian::PDF( Nwi );
        float BRDF = Lambertian::BRDF();
        Ray newRay = { P + wi * 1e-3, wi };
        ambient = ambient + BRDF / PDF * traceRay( newRay, nextDepth - 1, throughput * 0.8f ).color * Nwi;
    }
    return ambient * ambientOcclusion / (float) numAmbientSamples;
}

CanvasData RayTracer::traceRay( Ray& ray, int nextDepth, float throughput ) {
    IntersectionData cIData = closestIntersection( ray );
    if ( cIData.t == __FLT_MAX__ ) return { BACKGROUND_COLOR * throughput, { 0, 0, 0 }, BACKGROUND_COLOR * throughput };
    Vector3f P = ray.origin + ray.direction * cIData.t;
    RGB materialColor;
    Material material;
    float ambientOcclusion;
    float roughness;
    if ( cIData.triangle != nullptr ) {
        cIData.N = cIData.triangle->getNormal( P );
        materialColor = cIData.triangle->getColor( P );
        material = cIData.triangle->getMaterial();
        ambientOcclusion = cIData.triangle->getAmbient( P ).r;
        roughness = cIData.triangle->getRoughness( P );
    } else {
        cIData.N = cIData.sphere->getNormal( P );
        materialColor = cIData.sphere->getColor( P );
        material = cIData.sphere->material;
        ambientOcclusion = cIData.sphere->getAmbient( P ).r;
        roughness = cIData.sphere->getRoughness( P );
    }
    Vector3f vectorColor = ( cIData.N + Vector3f( 1, 1, 1 ) ) * 255 / 2;
    RGB normalColor = { vectorColor.x, vectorColor.y, vectorColor.z };
    if ( material.getIntensity() != 0 ) return { materialColor, normalColor, materialColor };
    RGB i = computeDiffuseLight( P, ray.direction * (-1), cIData );
    RGB diffuse = {};
    diffuse = { materialColor.r * i.r * throughput, materialColor.g * i.g * throughput, materialColor.b * i.b * throughput } ;
    if ( nextDepth == 0 ) return { diffuse, normalColor, materialColor };
    //Global illumination
    RGB ambient = computeAmbientLight( ray, cIData, roughness, ambientOcclusion, throughput, nextDepth );
    return { diffuse +  materialColor / 255 * ambient, normalColor, materialColor };
}

void RayTracer::load( Camera* c, Scene* s, Canvas* _canvas, int _depth, int _numAmbientSamples, int _numLightSamples ) {
    camera = Kokkos::View<Camera*>("camera");
    Kokkos::deep_copy(camera, *c);
    scene = Kokkos::View<Scene*>("scene");
    Kokkos::deep_copy(scene, *s);
    canvas = Kokkos::View<Canvas*>("canvas");
    Kokkos::deep_copy(canvas, *_canvas);
    bvh = new BVH( s->getTriangles(), s->getSpheres() );
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
            CanvasData colorData = traceRay( ray, depth, 1 );
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
//        Ray ray;
//        if ( sample == 0 ) {
//            ray = rayTracer->getCamera()->getPrimaryRay(i, j);
//        } else {
//            ray = rayTracer->getCamera()->getSecondaryRay(i, j);
//        }
//        colors(i, j) = colors(i, j) + rayTracer->traceRay(ray, rayTracer->getDepth(), 1) / (float) numSamples;
//    }
    //printf("array(%d, %d) = %f\n", i, j, colors(i,j).r);
    Ray ray = rayTracer->getCamera()->getPrimaryRay(i, j);
    CanvasData data = rayTracer->traceRay(ray, rayTracer->getDepth(), 1);
    colors(i, j) = data.color;
    normals(i, j) = data.normal;
    albedos(i, j) = data.albedo;
}
