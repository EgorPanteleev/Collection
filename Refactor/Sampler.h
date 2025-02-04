//
// Created by auser on 8/23/24.
//

#ifndef COLLECTION_SAMPLER_H
#define COLLECTION_SAMPLER_H
#include "Vec3.h"
#include "Vec2.h"
#include "Mat3.h"
#include "SystemUtils.h"
#include "Utils.h"
#include <cmath>

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

    DEVICE double PDF( double Nwi );

    DEVICE double PDF( const Vec3d& N, const Vec3d& wi );
}

namespace LambertianSampler {
    DEVICE double BRDF();

    DEVICE Vec3d getIncidentDir( const Vec3d& N, hiprandState& state ) {
        return CosineWeighted::getSample( N, state );
    }

    DEVICE double PDF( double Nwi );

    DEVICE double PDF( const Vec3d& N, const Vec3d& wi );
}

//we can avoid trigonometric functions for better perfomance ( but not now )

namespace OrenNayar {
    DEVICE double BRDF( const Vec3d& wi, const Vec3d& wo, double alpha );

    DEVICE Vec3d getIncidentDir( const Vec3d& N, hiprandState& state );

    DEVICE double PDF( double Nwi );

    DEVICE double PDF( const Vec3d& N, const Vec3d& wi );
}

namespace GGX {
    DEVICE double D( const Vec3d& m, const Vec2d& a );

    DEVICE double F( double dmwo );

    DEVICE double Lambda( const Vec3d& wo, const Vec2d& a );

    DEVICE double SmithG1( const Vec3d& wo, const Vec2d& a );

    DEVICE double DV( const Vec3d& m, const Vec3d& wo, const Vec2d& a );

    DEVICE double SmithG2( const Vec3d& wi, const Vec3d& wo, const Vec2d& a );

    DEVICE double BRDF( const Vec3d& wi, const Vec3d& wo, const Vec2d& a, double& PDF );

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
