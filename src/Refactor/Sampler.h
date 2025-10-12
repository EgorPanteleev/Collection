//
// Created by auser on 8/23/24.
//

#ifndef COLLECTION_SAMPLER_H
#define COLLECTION_SAMPLER_H
#include <cmath>
#include "Vec3.h"
#include "Vec2.h"
#include "Mat3.h"
#include "SystemUtils.h"
#include "Utils.h"
#include "Interval.h"


#define pd 1.0 // between 0 and 1, how much diffuse light be diffuse

Mat3d getTBN( const Vec3d& wo, const Vec3d& N );

namespace CosineWeighted {
    DEVICE Vec3d getSample( const Vec3d& N, hiprandState& state )  {
        double u = randomDouble( 0, 1, state );
        double v = randomDouble( 0, 1, state );
        double r = std::sqrt(u);
        double azimuth = v * 2 * M_PI;
        return { r * std::cos(azimuth), r * std::sin(azimuth), std::sqrt(1 - u) };
    }

    inline DEVICE double PDF( double Nwi ) {
        return cos( Nwi ) * M_1_PI;
    }

    inline DEVICE double PDF( const Vec3d& N, const Vec3d& wi ) {
        return PDF( dot( N, wi ) );
    }
}

namespace LambertianSampler {
    inline DEVICE double BRDF() {
        return pd * M_1_PI;
    }

    inline DEVICE Vec3d getIncidentDir( const Vec3d& N, hiprandState& state ) {
        return CosineWeighted::getSample( N, state );
    }

    inline DEVICE double PDF( double Nwi ) {
        return CosineWeighted::PDF( Nwi );
    }

    inline DEVICE double PDF( const Vec3d& N, const Vec3d& wi ) {
        return CosineWeighted::PDF( N, wi );
    }
}

//we can avoid trigonometric functions for better perfomance ( but not now )

namespace OrenNayarSampler {
    DEVICE double BRDF( const Vec3d& wi, const Vec3d& wo, double alpha ) {
        double A = 1 - ( alpha / ( 2 * ( alpha + 0.33 ) ) ); // A and B precomputed
        double B = ( 0.45 * alpha ) / ( alpha + 0.09 );
        //TODO
        Interval interval( 0.0, 1.0 );
        double cos_delta_phi = interval.clamp( (wi[0] * wo[0] + wi[1] * wo[1]) /
                                               std::sqrt((pow2(wi[0]) + pow2(wi[1])) *
                                                         (pow2(wo[0]) + pow2(wo[1]))) );

        // D = sin(alpha) * tan(beta), i[2] = dot(i, (0,0,1))
        double D = std::sqrt((1.0 - pow2(wi[2])) * (1.0 - pow2(wo[2]))) / std::max(wi[2], wo[2]);

        // A and B are pre-computed in constructor.
        return LambertianSampler::BRDF() * (A + B * cos_delta_phi * D);
    }

    inline DEVICE Vec3d getIncidentDir( const Vec3d& N, hiprandState& state ) {
        return LambertianSampler::getIncidentDir( N, state ); // move to cosine namespace
    }

    inline DEVICE double PDF( double Nwi ) {
        return LambertianSampler::PDF( Nwi );
    }

    inline DEVICE double PDF( const Vec3d& N, const Vec3d& wi ) {
        return LambertianSampler::PDF( N, wi );
    }
}

namespace GGX {
    inline DEVICE double D( const Vec3d& m, const Vec2d& a ) {
        return 1.0 / ( M_PI * a[0] * a[1] * pow2( pow2( m[0] / a[0] ) + pow2( m[1] / a[1] ) + pow2( m[2] ) ) );
    }

    inline DEVICE double F( double dmwo, double refractionIndex ) {
        double F0 = ( 1 - refractionIndex ) / ( 1 + refractionIndex );
        F0 = pow2( F0 );
        return F0 + (1.0 - F0) * pow(1.0 - dmwo, 5.0);
    }

    inline DEVICE double Lambda( const Vec3d& wo, const Vec2d& a ) {
        return ( -1.0 + std::sqrt( 1.0 + ( pow2 (a[0] * wo[0]) + pow2 (a[1] * wo[1]) ) / ( pow2(wo[2]) ) ) ) / 2.0;
    }

    inline DEVICE double SmithG1( const Vec3d& wo, const Vec2d& a ) {
        return 1.0 / (1.0 + Lambda(wo, a) );
    }

    inline DEVICE double DV( const Vec3d& m, const Vec3d& wo, const Vec2d& a ) {
        return SmithG1(wo, a) * dot(wo, m) * D(m, a) / wo[2];
    }

    inline DEVICE double SmithG2( const Vec3d& wi, const Vec3d& wo, const Vec2d& a ) {
        return 1.0 / (1.0 + Lambda(wo, a) + Lambda(wi, a));
    }

    inline DEVICE double BRDF( const Vec3d& wi, const Vec3d& wo, const Vec2d& a, double& PDF, double refractionIndex ) {
        Vec3d m = (wo + wi).normalize();
        PDF = DV(m, wo, a) / ( 4.0 * dot(m, wo));
        return F( dot( m, wo ), refractionIndex ) * D(m, a) * SmithG2(wi, wo, a) / ( 4.0 * wo[2] * wi[2]);
    }

    DEVICE Vec3d getNormal( const Vec3d& wo, const Vec2d& a, hiprandState& state ) {
        double u = randomDouble( 0, 1, state );
        double v = randomDouble( 0, 1, state );
        Vec3d Vh = (Vec3d(a[0] * wo[0], a[1] * wo[1], wo[2])).normalize();

        // orthonormal basis (with special case if cross product is zero)
        double len2 = pow2( Vh[0] ) + pow2( Vh[1] );
        Vec3d T1 = len2 > 0.0 ? Vec3d( -Vh[1], Vh[0], 0.0 ) * 1.0 / std::sqrt(len2) : Vec3d(1.0, 0.0, 0.0);
        Vec3d T2 = cross( Vh, T1 );

        // parameterization of the projected area
        double r = std::sqrt( u );
        double phi = v * 2 * M_PI;
        double t1 = r * std::cos(phi);
        double t2 = r * std::sin(phi);
        double s = 0.5 * ( 1.0 + Vh[2] );
        t2 = ( 1.0 - s ) * std::sqrt(1.0 - pow2( t1 ) ) + s * t2;

        // reprojection onto hemisphere
        Vec3d Nh = t1 * T1 + t2 * T2 + std::sqrt(std::max( 0.0, 1.0 - pow2(t1) - pow2(t2))) * Vh;

        // transforming the normal back to the ellipsoid configuration
        return ( Vec3d(a[0] * Nh[0], a[1] * Nh[1], std::max(0.0, Nh[2])) ).normalize();
    }
}

#endif //COLLECTION_SAMPLER_H
