#include "RayTracer.h"
#include <cmath>
#include "Utils.h"
#include "Sheduler.h"
//#define BACKGROUND_COLOR RGB(0, 0, 0)
#define BACKGROUND_COLOR RGB(173, 216, 230)
//#define BACKGROUND_COLOR RGB(255, 255, 255)

RayTracer::RayTracer( Camera* c, Scene* s, Canvas* _canvas, int _depth, int _numAmbientSamples, int _numLightSamples  ):
depth( _depth ), numAmbientSamples( _numAmbientSamples ), numLightSamples( _numLightSamples ) {
    camera = Kokkos::View<Camera*>("camera");
    Kokkos::deep_copy(camera, *c);
    s->fillTriangles();
    scene = Kokkos::View<Scene*>("scene");
    Kokkos::deep_copy(scene, *s);
    canvas = Kokkos::View<Canvas*>("canvas");
    Kokkos::deep_copy(canvas, *_canvas);
    bvh = new BVH( s->getTriangles() );
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
//        if ( iData.t == std::numeric_limits<float>::max()) continue;
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


float RayTracer::computeLight( const Vector3f& P, const Vector3f& V, const IntersectionData& iData ) {
    float i = 0;
    Vector3f N = iData.N;
    float d = 0.8;
    float s = 1 - d;
    for ( auto light: scene(0).lights ) {
        int numLS = ( light->getType() == Light::Type::POINT ) ? 1 : numLightSamples;
        float i1 = 0;
        if ( light->isIntersectsWithRay( Ray( P, V ) ) ) return -light->intensity;
        for ( size_t j = 0; j < numLS; j++ ) {
            Vector3f origin = light->getSamplePoint();
            Vector3f L = (origin - P).normalize();
            Ray ray = Ray(origin, L * ( -1 ) );
            IntersectionData cIData = closestIntersection( ray );
            if (cIData.triangle == nullptr) continue;
            if (cIData.triangle != iData.triangle) continue;
            float dNL = dot(N, L);
            i1 += d * light->intensity * dNL;
            //specular
            //if (iData.triangle->owner->getMaterial().getDiffuse() == -1) continue;

            Vector3f H = (L + V ).normalize();
            float roughness = 0.1;
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
            float refraction = 0.9;
            float F0 = pow( refraction - 1, 2 ) / pow( refraction + 1, 2 );
            float F = F0 + ( 1 - F0 ) * pow( 1 - dVH, 5 );


            float rs = ( D * G * F ) / ( 4 * dNL * dNV );

            i1 += s * rs * light->intensity * dNL;

//            Vector3f R = (N * 2 * dNL - L).normalize();
//            float dRV = dot(R, V.normalize());
//            if (dRV > 0) i1 += light->intensity * pow(dRV, iData.triangle->owner->getMaterial().getDiffuse());
        }
        i1 /= numLS;
        i += i1;
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

RGB RayTracer::traceRay( Ray& ray, int nextDepth, float throughput ) {
    IntersectionData cIData = closestIntersection( ray );
    if ( cIData.t == std::numeric_limits<float>::max() ) return BACKGROUND_COLOR * throughput;
    Vector3f P = ray.origin + ray.direction * cIData.t;
    float i = computeLight( P, ray.direction * (-1), cIData );
    if ( i < 0 ) return RGB( 255, 255, 255 ) * (-i);
    RGB diffuse = cIData.triangle->owner->getMaterial().getColor() * i * throughput;
    if ( nextDepth == 0 ) return diffuse;
    //REFLECTION
    float r = cIData.triangle->owner->getMaterial().getReflection();// -> shining
    if ( r == 0 ) {
        //Global illumination
        RGB ambient = {};
        throughput *= 0.8;
        for (int j = 0; j < numAmbientSamples; j++ ) {
            Vector3f samplePoint = generateSamplePoint( cIData.N );
            Ray sampleRay = { P + samplePoint * 1e-3, samplePoint };
            ambient = ambient + traceRay( sampleRay, nextDepth - 1, throughput ) * dot( samplePoint, cIData.N );
        }
        //default
        //ambient = ambient * 2 / numSamples;
        //cosine
        ambient = ambient * 1 / (float) numAmbientSamples;
        return diffuse + ambient;
    }
    else {
        Vector3f N = cIData.N.normalize();
        Vector3f reflectedDir = ( ray.direction - N * 2 * dot(N, ray.direction ) );
        Ray reflectedRay( P + reflectedDir * 1e-3, reflectedDir );
        RGB reflectedColor = traceRay( reflectedRay, nextDepth - 1, throughput );
        return diffuse * ( 1 - r ) + reflectedColor * r;
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

void RayTracer::traceAllRays( Type type ) {
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
            diffuse = { 0, 0, 0 };
            ambient = { 0, 0, 0 };
            specular = { 0, 0, 0 };
            RGB color = traceRay( ray, depth, 1 );
            color.scaleTo( 255 );
            canvas(0).setPixel( x, y, color );
        }
    }
}
void RayTracer::traceAllRaysParallel() {
    float uX = camera(0).Vx / canvas(0).getW();
    float uY = camera(0).Vy / canvas(0).getH();
    float uX2 = uX / 2.0f;
    float uY2 = uY / 2.0f;
    Vector3f from = camera(0).origin;
    float Vx2 = camera(0).Vx / 2;
    float Vy2 = camera(0).Vy / 2;
    Kokkos::View<RGB**> result = Kokkos::View<RGB**>("colors", canvas(0).getW(), canvas(0).getH() );
    RenderFunctor renderFunctor( uX, uY, uX2, uY2, Vx2, Vy2, camera(0).dV, depth, from, this, result );
    typedef Kokkos::MDRangePolicy<Kokkos::Rank<2>> range_policy_2d;
    range_policy_2d policy({0, 0}, {canvas(0).getW(), canvas(0).getH()});
    Kokkos::parallel_for("parallel2D", policy, renderFunctor);
    for ( int i = 0; i < canvas(0).getW(); i++ ) {
        for ( int j = 0; j < canvas(0).getH(); j++ ) {
            canvas(0).setPixel( i, j, result(i, j) );
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


RenderFunctor::RenderFunctor(float _uX, float _uY, float _uX2, float _uY2, float _Vx2,
                             float _Vy2, float _dV, int _depth, Vector3f _from, RayTracer* _rayTracer, Kokkos::View<RGB**>& result )
        : uX(_uX), uY(_uY), uX2(_uX2), uY2(_uY2), Vx2(_Vx2),
          Vy2(_Vy2), dV(_dV), from(_from), rayTracer( _rayTracer ), depth( _depth ), colors( result ) {
}

void RenderFunctor::operator()(const int i, const int j) const {
    Vector3f dir = { -Vx2 + uX2 + i * uX, -Vy2 + uY2 + j * uY, dV  };
    Ray ray( from, dir);
    colors(i, j) = rayTracer->traceRay( ray, depth, 1 );
    //printf("array(%d, %d) = %f\n", i, j, colors(i,j).r);
}
