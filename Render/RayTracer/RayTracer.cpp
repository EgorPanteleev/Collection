#include "RayTracer.h"
#include <cmath>
#include "Utils.h"
//#include "Sheduler.h"
#define BACKGROUND_COLOR RGB(0, 0, 0)
//#define BACKGROUND_COLOR RGB(255, 0, 0)
//#define BACKGROUND_COLOR RGB(255, 255, 255)

RayTracer::RayTracer( Camera* c, Scene* s, Canvas* _canvas, int _depth, int _numAmbientSamples, int _numLightSamples  ):
depth( _depth ), numAmbientSamples( _numAmbientSamples ), numLightSamples( _numLightSamples ) {
    camera = Kokkos::View<Camera*>("camera");
    Kokkos::deep_copy(camera, *c);
    scene = Kokkos::View<Scene*>("scene");
    Kokkos::deep_copy(scene, *s);
    canvas = Kokkos::View<Canvas*>("canvas");
    Kokkos::deep_copy(canvas, *_canvas);
    bvh = new BVH( s->getTriangles(), s->getSpheres() );
    bvh->BuildBVH();
//    bvh = Kokkos::View<BVH*>("BVH");
//    Kokkos::deep_copy(bvh, *_bvh);
}
RayTracer::~RayTracer() {
    //delete canvas;
}

//closestIntersectionData RayTracer::closestIntersection( Ray& ray ) {
//    closestIntersectionData cIData;
//    for ( const auto& object: scene->objects ) {
//        IntersectionData iData = object->intersectsWithRay(ray);
//        if ( iData.t == __FLT_MAX__) continue;
//        if ( iData.t <= 0.05 ) continue;
//        if ( cIData.t < iData.t ) continue;
//        cIData.t = iData.t;
//        cIData.N = iData.N;
//        cIData.object = object;
//    }
//    return cIData;
//}


IntersectionData RayTracer::closestIntersection( Ray& ray ) {
    return bvh->IntersectBVH( ray, 0 );
}

template <typename Type>
float getIntensity( Type* light ) {
    return light->getIntensity();
}


RGB RayTracer::computeDiffuseLight( const Vector3f& P, const Vector3f& V, const IntersectionData& iData ) {
    RGB i;
    Vector3f N = iData.N;
    float d = 0.8;
    float s = 1 - d;
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
            i1 = i1 + d * light->getIntensity() * ( light->getColor() * del ) * dNL * inverseDistance2;
            //specular
            //if (iData.triangle->owner->getMaterial().getDiffuse() == -1) continue;

            Vector3f H = (L + V ).normalize();
            //float power = 1 / roughness;
            //float power = 2.0f / pow( roughness, 4 ) - 2; //better
            auto a2 = (float) pow( roughness, 4 );
            float dNH = dot( N, H );

            auto X = []( float val ) {
                return ( val > 0 ) ? 1 : 0;
            };


            //float D = 1.0f / ( M_PI * a2 ) * pow( dNH, power ); //BLINN

            float tan = ( dNH * dNH - 1 ) / ( dNH * dNH );
            //float D = 1.0f / ( M_PI * a2 ) * exp( tan / a2 ) / pow( dNH, 4 ); //BECKMANN

            float D = ( a2 * (float) X( dNH ) ) / (float) ( M_PI * pow( ( dNH * dNH * ( a2 - tan ) ), 2 ) ); //GGX

            float dNV = dot( N, V );

            float dVH = dot( V, H ); // really H, ost M

            //float G = std::min( 1.0f, std::min(  2 * dNH * dNV / dVH, 2 * dNH * dNL / dVH  ) );  //cook-torrance

            auto G1 = [=]( const Vector3f& x ) {
                float dNX = dot(x,N);
                return X(dot(x,H)/dNX) * 2 / ( 1 + sqrt( 1 - a2 * ( dNX * dNX - 1 ) / ( dNX * dNX ) ) );
            };

            float G = G1( V ) * G1( L ); //GGX
            float refraction = 0.5;
            float F0 = pow( refraction - 1, 2 ) / pow( refraction + 1, 2 );
            float F = F0 + ( 1 - F0 ) * pow( 1 - dVH, 5 );

            float rs = ( D * G * F ) / ( 4 * dNL * dNV );
            //TODO idk about dist
            i1 = i1 + s * rs * light->getIntensity() * ( light->getColor() * del ) * dNL * inverseDistance2;
//            Vector3f R = (N * 2 * dNL - L).normalize();
//            float dRV = dot(R, V.normalize());
//            if (dRV > 0) i1 += light->intensity * pow(dRV, iData.triangle->owner->getMaterial().getDiffuse());
        }
        i = i + i1 / (float) numLightSamples;
    }
    //if ( i > 1 ) i = 1;
    return i;
}


float RayTracer::computeLight( const Vector3f& P, const Vector3f& V, const IntersectionData& iData ) {
    float i = 0;
    Vector3f N = iData.N;
    float d = 0.8;
    float s = 1 - d;
    for ( auto light: scene(0).getLights() ) {
        //int numLS = ( light->getType() == Light::Type::POINT ) ? 1 : numLightSamples;
        float i1 = 0;
        for ( size_t j = 0; j < numLightSamples; j++ ) {
            Vector3f origin = light->getSamplePoint();
            Vector3f L = (origin - P).normalize();
            Ray ray = Ray(origin, L * ( -1 ) );
            IntersectionData cIData = closestIntersection( ray );
            if (cIData.triangle == nullptr && cIData.sphere == nullptr ) continue;
            if (cIData.triangle != iData.triangle ) continue;
            if (cIData.sphere != iData.sphere ) continue;
            float dNL = dot( iData.N, L);
            if ( dNL < 0 ) continue;
            float distance = cIData.t * 0.01;
            i1 += d * light->getIntensity() * dNL / ( distance * distance );
            //specular
            //if (iData.triangle->owner->getMaterial().getDiffuse() == -1) continue;

            Vector3f H = (L + V ).normalize();
            float roughness = 0.5;
            //float power = 1 / roughness;
            float power = 2.0f / pow( roughness, 4 ) - 2; //better
            float a2 = pow( roughness, 4 );
            float dNH = dot( N, H );

            auto X = []( float val ) {
                return ( val > 0 ) ? 1 : 0;
            };


            //float D = 1.0f / ( M_PI * a2 ) * pow( dNH, power ); //BLINN

            float tan = ( dNH * dNH - 1 ) / ( dNH * dNH );
            //float D = 1.0f / ( M_PI * a2 ) * exp( tan / a2 ) / pow( dNH, 4 ); //BECKMANN

            float D = ( a2 * X( dNH ) ) / ( M_PI * pow( ( dNH * dNH * ( a2 - tan ) ), 2 ) ); //GGX

            float dNV = dot( N, V );

            float dVH = dot( V, H ); // really H, ost M

            //float G = std::min( 1.0f, std::min(  2 * dNH * dNV / dVH, 2 * dNH * dNL / dVH  ) );  //cook-torrance

            auto G1 = [=]( const Vector3f& x ) {
                float dNX = dot(x,N);
                return X(dot(x,H)/dNX) * 2 / ( 1 + sqrt( 1 - a2 * ( dNX * dNX - 1 ) / ( dNX * dNX ) ) );
            };

            float G = G1( V ) * G1( L ); //GGX
            float refraction = 0.5;
            float F0 = pow( refraction - 1, 2 ) / pow( refraction + 1, 2 );
            float F = F0 + ( 1 - F0 ) * pow( 1 - dVH, 5 );

            float rs = ( D * G * F ) / ( 4 * dNL * dNV );
            //TODO idk about dist
            i1 += s * rs * light->getIntensity() * dNL / ( distance * distance );
//            Vector3f R = (N * 2 * dNL - L).normalize();
//            float dRV = dot(R, V.normalize());
//            if (dRV > 0) i1 += light->intensity * pow(dRV, iData.triangle->owner->getMaterial().getDiffuse());
        }
        i += i1 / (float) numLightSamples;
    }
    //if ( i > 1 ) i = 1;
    return i;
}
//default
//Vector3f generateSamplePoint( Vector3f N ) {
//    const float u = rand() / (float) RAND_MAX;
//    const float v = rand() / (float) RAND_MAX;
//    float r = sqrt( 1 - u * u );
//    float theta = 2.0 * M_PI * v;
//    Vector3f up = { 0, 0, 1 };
//    float angle = acos(dot(up, N)) * 180 * M_1_PI;
//    Vector3f axis = up.cross( N );
//    Vector3f res = { r * (float) cos(theta), r * (float) sin(theta), u };
//    if ( up == N ) return res;
//    else if ( up == N * ( -1 ) ) return res * ( -1 );
//    res = Mat3f::getRotationMatrix( axis, angle ) * res;
//    return res;
//}
////cosine
Vector3f generateSamplePoint( Vector3f N ) {
    const float u = rand() / (float) RAND_MAX;
    const float v = rand() / (float) RAND_MAX;
    float r = sqrt( u );
    float theta = 2.0 * M_PI * v;
    Vector3f up = { 0, 0, 1 };
    float angle = acos(dot(up, N)) * 180 * M_1_PI;
    Vector3f axis = up.cross( N );
    float x = r * (float) cos(theta);
    float y = r * (float) sin(theta);
    float z = std::sqrt( 1.0f - x * x - y * y );
    Vector3f res = { x, y, z };
    if ( up == N ) return res;
    else if ( up == N * ( -1 ) ) return res * ( -1 );
    res = Mat3f::getRotationMatrix( axis, angle ) * res;
    return res;
}

CanvasData RayTracer::traceRay( Ray& ray, int nextDepth, float throughput ) {
    IntersectionData cIData = closestIntersection( ray );
    if ( cIData.t == __FLT_MAX__ ) return { BACKGROUND_COLOR * throughput, { 0, 0, 0 }, BACKGROUND_COLOR * throughput };
    Vector3f P = ray.origin + ray.direction * cIData.t;
    RGB materialColor;
    Material material;
    float ambientOcclusion;
    if ( cIData.triangle != nullptr ) {
        cIData.N = cIData.triangle->getNormal( P );
        materialColor = cIData.triangle->getColor( P );
        material = cIData.triangle->owner->getMaterial();
        ambientOcclusion = cIData.triangle->getAmbient( P ).r;
    } else {
        cIData.N = cIData.sphere->getNormal( P );
        materialColor = cIData.sphere->getColor( P );
        material = cIData.sphere->material;
        ambientOcclusion = cIData.sphere->getAmbient( P ).r;
    }
    Vector3f vectorColor = ( cIData.N + Vector3f( 1, 1, 1 ) ) * 255 / 2;
    RGB normalColor = { vectorColor.x, vectorColor.y, vectorColor.z };
    if ( material.getIntensity() != 0 ) return { materialColor, normalColor, materialColor };
    RGB i = computeDiffuseLight( P, ray.direction * (-1), cIData );
    //if ( i < 0 ) return RGB( 255, 255, 255 ) * (-i);
    RGB diffuse = {};
    float reflection;
    diffuse = { materialColor.r * i.r * throughput, materialColor.g * i.g * throughput, materialColor.b * i.b * throughput } ;
    reflection = material.getReflection();// -> shining

    if ( nextDepth == 0 ) return { diffuse, normalColor, materialColor };
    //REFLECTION
    if ( reflection == 0 ) {
        //Global illumination
        RGB ambient = {};
        throughput *= 0.8;
        for (int j = 0; j < numAmbientSamples; j++ ) {
            Vector3f samplePoint = generateSamplePoint( cIData.N );
            Ray sampleRay = { P + samplePoint * 1e-3, samplePoint };
            ambient = ambient + traceRay( sampleRay, nextDepth - 1, throughput ).color * dot( samplePoint, cIData.N );
        }
        //default
        //ambient = ambient * 2 / numSamples;
        //cosine
        ambient = ambient * ambientOcclusion / (float) numAmbientSamples;
        return { diffuse + ambient, normalColor, materialColor };
    }
    else {
        Vector3f N = cIData.N;
        Vector3f reflectedDir = ( ray.direction - N * 2 * dot(N, ray.direction ) );
        Ray reflectedRay( P + reflectedDir * 1e-3, reflectedDir );
        CanvasData reflectedData = traceRay( reflectedRay, nextDepth - 1, throughput );
        return { diffuse * ( 1 - reflection ) + reflectedData.color * reflection, reflectedData.normal, materialColor };
    }

    //r = ambient + sum(light_color * dot(N,L) * ( d * diffuse + s * specular ) )
    // ambient - the same
    // diffuse - the same
    // specular -
    //



//    float3 wi = material->Sample(wo, normal, sampler);
//    float pdf = material->Pdf(wi, normal);
//
//    // Accumulate the brdf attenuation
//    throughput = throughput * material->Eval(wi, wo, normal) / pdf;

//    void LambertBRDF::Sample(float3 outputDirection, float3 normal, UniformSampler *sampler) {
//        float rand = sampler->NextFloat();
//        float r = std::sqrtf(rand);
//        float theta = sampler->NextFloat() * 2.0f * M_PI;
//
//        float x = r * std::cosf(theta);
//        float y = r * std::sinf(theta);
//
//        // Project z up to the unit hemisphere
//        float z = std::sqrtf(1.0f - x * x - y * y);
//
//        return normalize(TransformToWorld(x, y, z, normal));
//    }

//    float LambertBRDF::Pdf(float3 inputDirection, float3 normal) {
//        return dot(inputDirection, normal) * M_1_PI;
//    }

//    float3 LambertBRDF::Eval(float3 inputDirection, float3 outputDirection, float3 normal) const override {
//        return m_albedo * M_1_PI * dot(inputDirection, normal);
//    }

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
    float uX = camera(0).Vx / canvas(0).getW();
    float uY = camera(0).Vy / canvas(0).getH();
    float uX2 = uX / 2.0f;
    float uY2 = uY / 2.0f;
    Vector3f from = camera(0).origin;
    float Vx2 = camera(0).Vx / 2;
    float Vy2 = camera(0).Vy / 2;
    for ( int x = 0; x < canvas(0).getW(); ++x ) {
        printProgress( x );
        for ( int y = 0; y < canvas(0).getH(); ++y ) {
            Vector3f dir = { -Vx2 + uX2 + x * uX, -Vy2 + uY2 + y * uY, camera(0).dV  };
            Ray ray( from, dir);
            if ( x != 1600 ) continue;
            if ( y != 1000 ) continue;
            //diffuse = { 0, 0, 0 };
            //ambient = { 0, 0, 0 };
            //specular = { 0, 0, 0 };
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
