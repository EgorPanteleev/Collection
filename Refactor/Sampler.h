//
// Created by auser on 8/23/24.
//

#ifndef COLLECTION_SAMPLER_H
#define COLLECTION_SAMPLER_H
#include "Vec3.h"
#include "Vec2.h"
#include "Mat3.h"
#include "SystemUtils.h"

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

    DEVICE Vec3d getNormal( const Vec3d& wo, const Vec2d& a, hiprandState& state );
}

#endif //COLLECTION_SAMPLER_H
