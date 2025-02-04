//
// Created by auser on 11/26/24.
//

#ifndef COLLECTION_MATERIAL_H
#define COLLECTION_MATERIAL_H
#include "Utils.h"
#include "Ray.h"
#include "HitRecord.h"
#include "SystemUtils.h"
#include "Texture.h"
#include "CoordinateSystem.h"
#include "Sampler.h"



class Material {
public:
    enum Type {
        LAMBERTIAN,
        METAL,
        DIELECTRIC,
        LIGHT,
        UNKNOWN
    };
    HOST_DEVICE Material(): type( UNKNOWN ) {}
    HOST_DEVICE Material( Type type ): type(type) {}
    // virtual HOST_DEVICE bool scatter( const Ray& rayIn, const HitRecord& hitRecord, RGB& attenuation, Ray& scattered ) const = 0;
#if HIP_ENABLED
    virtual HOST Material* copyToDevice() = 0;

    virtual HOST Material* copyToHost() = 0;

    virtual HOST void deallocateOnDevice() = 0;
#endif

    Type type;
};


//        Vec3d wi_T = Lambertian::getIncidentDir( traceData.cs.getNormal() );
//        Vec3d wi = traceData.cs.from( wi_T );
//        double Nwi = saturate(wi_T[2]);
//        double PDF = Lambertian::PDF( Nwi );
//        double BRDF = Lambertian::BRDF();
//        Ray newRay = { traceData.P + wi * 1e-3, wi };
//        CanvasData data;
//        traceRay( newRay, data, nextDepth - 1 );
//        ambient += BRDF / PDF * data.color * Nwi;


class Lambertian: public Material {
public:
//    Lambertian(): Material( LAMBERTIAN ), texture() {}
//    Lambertian( const RGB& albedo ): Material( LAMBERTIAN ), texture( new SolidColor( albedo ) ) {}
//    Lambertian( Texture* texture ): Material( LAMBERTIAN ), texture( texture ) {}
    Lambertian(): Material( LAMBERTIAN ), albedo() {}
    Lambertian( const RGB& albedo ): Material( LAMBERTIAN ), albedo( albedo ) {}
    DEVICE bool scatter( const Ray& rayIn, const HitRecord& hitRecord, RGB& attenuation, Ray& scattered, hiprandState& state ) const {
        CoordinateSystem cs( hitRecord.N );
        Vec3d wi_T = LambertianSampler::getIncidentDir( cs.getNormal(), state );
        Vec3d wi = cs.from( wi_T );
        scattered = { hitRecord.p + wi * 1e-3, wi };
        double Nwi = saturate( wi_T[2] );
        double PDF = LambertianSampler::PDF( Nwi );
        double BRDF = LambertianSampler::BRDF();
        attenuation = BRDF / PDF * albedo * Nwi;//value( texture, hitRecord.u, hitRecord.v, hitRecord.p );
        return true;
    }
#if HIP_ENABLED
    HOST Material* copyToDevice() override {
        auto device = HIP::allocateOnDevice<Lambertian>();
        HIP::copyToDevice( this, device );
        //device->texture = texture->copyToDevice();
        return device;
    }

    HOST Material* copyToHost() override {
        auto host = new Lambertian();
        HIP::copyToHost( host, this );
       // host->texture = texture->copyToHost();
        HIP::deallocateOnDevice( this );
        return host;
    }

    HOST void deallocateOnDevice() override {
        //texture->deallocateOnDevice();
        HIP::deallocateOnDevice<Lambertian>( this );
    }
#endif
public:
    RGB albedo;
    //Texture* texture;
};

class Metal: public Material {
public:
    Metal(): Material( METAL ), albedo(), fuzz() {}
    HOST_DEVICE Metal( const RGB& albedo, double fuzz ): Material( METAL ), albedo( albedo ), fuzz( fuzz ) {}
    DEVICE bool scatter( const Ray& rayIn, const HitRecord& hitRecord, RGB& attenuation, Ray& scattered, hiprandState& state ) const {
        CoordinateSystem cs( hitRecord.N );
        Vec3d wo_T = cs.to( rayIn.direction * (-1) );
        Vec2d alpha = { pow2( fuzz ), pow2( fuzz ) };
        Vec3d H_T = GGX::getNormal( wo_T, alpha, state );
        Vec3d wi_T = reflect( wo_T, H_T ) * ( -1 );
        double Nwi = saturate(wi_T[2]);
        double PDF;
        double BRDF = GGX::BRDF( wi_T, wo_T, alpha, PDF, 0.0256 );
        Vec3d wi = cs.from( wi_T );
        scattered = { hitRecord.p + wi * 1e-3, wi };

        attenuation = BRDF / PDF * albedo * Nwi;;
        return true;
    }


#if HIP_ENABLED
    HOST Material* copyToDevice() override {
        auto device = HIP::allocateOnDevice<Metal>();
        HIP::copyToDevice( this, device );
        return device;
    }

    HOST Material* copyToHost() override {
        auto host = new Metal();
        HIP::copyToHost( host, this );
        HIP::deallocateOnDevice( this );
        return host;
    }

    HOST void deallocateOnDevice() override {
        HIP::deallocateOnDevice<Metal>( this );
    }
#endif

public:
    RGB albedo;
    double fuzz;
};

class Dielectric: public Material {
public:
    Dielectric(): Material( DIELECTRIC ), refractionIndex() {}
    Dielectric( double refractionIndex ):  Material( DIELECTRIC ), refractionIndex( refractionIndex ) {}
    HOST_DEVICE bool scatter( const Ray& rayIn, const HitRecord& hitRecord, RGB& attenuation, Ray& scattered,  hiprandState& state ) const {
        attenuation = 1;
        double ri = hitRecord.frontFace ? ( 1.0 / refractionIndex ) : refractionIndex;

        double cosTheta = std::min( dot( -rayIn.direction, hitRecord.N ), 1.0 );
        double sinTheta = std::sqrt( 1.0 - pow( cosTheta, 2 ) );

        bool cannotRefract = ri * sinTheta > 1;

        Vec3d direction;

        if ( cannotRefract || reflectance( cosTheta, ri ) > randomDouble( state ) )
            direction = reflect( rayIn.direction, hitRecord.N );
        else
            direction = refract( rayIn.direction, hitRecord.N, ri );

        scattered = { hitRecord.p, direction };
        return true;
    }

    HOST_DEVICE static double reflectance( double cosine, double refractionIndex ) {
        double r0 = ( 1 - refractionIndex ) / ( 1 + refractionIndex );
        r0 = pow( r0, 2 );
        return r0 + ( 1 - r0 ) * pow( 1 - cosine, 5 );
    }

#if HIP_ENABLED
    HOST Material* copyToDevice() override {
        auto device = HIP::allocateOnDevice<Dielectric>();
        HIP::copyToDevice( this, device );
        return device;
    }

    HOST Material* copyToHost() override {
        auto host = new Dielectric();
        HIP::copyToHost( host, this );
        HIP::deallocateOnDevice( this );
        return host;
    }

    HOST void deallocateOnDevice() override {
        HIP::deallocateOnDevice<Dielectric>( this );
    }
#endif

    double refractionIndex;
};

class Light: public Material {
public:
    Light(): Material( LIGHT ), albedo(), intensity() {}
    Light( double intensity ):  Material( LIGHT ), albedo(), intensity( intensity ) {}
    Light( const RGB& albedo ):  Material( LIGHT ), albedo(albedo), intensity() {}
    Light( const RGB& albedo, double intensity ):  Material( LIGHT ), albedo(albedo), intensity( intensity ) {}

    DEVICE RGB emit(double u, double v, const Point3d & p) const {
        return intensity * albedo;
    }

#if HIP_ENABLED
    HOST Material* copyToDevice() override {
        auto device = HIP::allocateOnDevice<Light>();
        HIP::copyToDevice( this, device );
        return device;
    }

    HOST Material* copyToHost() override {
        auto host = new Light();
        HIP::copyToHost( host, this );
        HIP::deallocateOnDevice( this );
        return host;
    }

    HOST void deallocateOnDevice() override {
        HIP::deallocateOnDevice<Light>( this );
    }
#endif
    RGB albedo;
    double intensity;
};

#endif //COLLECTION_MATERIAL_H
