//
// Created by auser on 11/26/24.
//

#ifndef COLLECTION_MATERIAL_H
#define COLLECTION_MATERIAL_H
#include "Utils.h"
#include "Ray.h"
#include "HittableList.h"
#include "RGB.h"

class Material {
public:
    virtual bool scatter( const Ray& rayIn, const HitRecord& hitRecord, RGB& attenuation, Ray& scattered ) const {
        return false;
    }
};

class Lambertian: public Material {
public:
    Lambertian( const RGB& albedo ): albedo( albedo ) {}
    bool scatter( const Ray& rayIn, const HitRecord& hitRecord, RGB& attenuation, Ray& scattered ) const override {
        Vec3d scatterDir = hitRecord.N + Vec3d( randomDouble( -1, 1 ), randomDouble( -1, 1 ), randomDouble( -1, 1 ) ).normalize();
        scattered = { hitRecord.p, scatterDir };
        attenuation = albedo;
        return true;
    }
private:
    RGB albedo;
};

class Metal: public Material {
public:
    Metal( const RGB& albedo, double fuzz ): albedo( albedo ), fuzz( fuzz ) {}
    bool scatter( const Ray& rayIn, const HitRecord& hitRecord, RGB& attenuation, Ray& scattered ) const override {
        Vec3d reflected = reflect( rayIn.direction, hitRecord.N ).normalize();
        reflected += fuzz * Vec3d( randomDouble( -1, 1 ), randomDouble( -1, 1 ), randomDouble( -1, 1 ) ).normalize();
        scattered = { hitRecord.p, reflected };
        attenuation = albedo;
        return true;
    }

private:
    RGB albedo;
    double fuzz;
};

class Dielectric: public Material {
public:
    Dielectric( double refractionIndex ): refractionIndex( refractionIndex ) {}
    bool scatter( const Ray& rayIn, const HitRecord& hitRecord, RGB& attenuation, Ray& scattered ) const override {
        attenuation = { 1, 1, 1 };
        double ri = hitRecord.frontFace ? ( 1.0 / refractionIndex ) : refractionIndex;

        double cosTheta = std::min( dot( -rayIn.direction, hitRecord.N ), 1.0 );
        double sinTheta = std::sqrt( 1.0 - pow( cosTheta, 2 ) );

        bool cannotRefract = ri * sinTheta > 1;

        Vec3d direction;

        if ( cannotRefract || reflectance( cosTheta, ri ) > randomDouble() )
            direction = reflect( rayIn.direction, hitRecord.N );
        else
            direction = refract( rayIn.direction, hitRecord.N, ri );

        scattered = { hitRecord.p, direction };
        return true;
    }

    static double reflectance( double cosine, double refractionIndex ) {
        double r0 = ( 1 - refractionIndex ) / ( 1 + refractionIndex );
        r0 = pow( r0, 2 );
        return r0 + ( 1 - r0 ) * pow( 1 - cosine, 5 );
    }

    double refractionIndex;
};

#endif //COLLECTION_MATERIAL_H
