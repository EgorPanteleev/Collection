//
// Created by auser on 8/23/24.
//

#ifndef COLLECTION_SAMPLER_H
#define COLLECTION_SAMPLER_H
#include "Vec3.h"
#include "Vec2.h"
#include "Mat3.h"

#include <cmath>

#define pd 1.0 // between 0 and 1, how much diffuse light be diffuse

Mat3d getTBN( const Vec3d& wo, const Vec3d& N );

namespace CosineWeighted {
    Vec3d getSample( const Vec3d& N );

    double PDF( double Nwi );

    double PDF( const Vec3d& N, const Vec3d& wi );
}

namespace Lambertian {
    double BRDF();

    Vec3d getIncidentDir( const Vec3d& N );

    double PDF( double Nwi );

    double PDF( const Vec3d& N, const Vec3d& wi );
}

//we can avoid trigonometric functions for better perfomance ( but not now )

namespace OrenNayar {
    double BRDF( const Vec3d& wi, const Vec3d& wo, double alpha );

    Vec3d getIncidentDir( const Vec3d& N );

    double PDF( double Nwi );

    double PDF( const Vec3d& N, const Vec3d& wi );
}

namespace GGX {
    double D( const Vec3d& m, const Vec2d& a );

    double F( double dmwo );

    double Lambda( const Vec3d& wo, const Vec2d& a );

    double SmithG1( const Vec3d& wo, const Vec2d& a );

    double DV( const Vec3d& m, const Vec3d& wo, const Vec2d& a );

    double SmithG2( const Vec3d& wi, const Vec3d& wo, const Vec2d& a );

    double BRDF( const Vec3d& wi, const Vec3d& wo, const Vec2d& a, double& PDF );

    Vec3d getNormal( const Vec3d& wo, const Vec2d& a );
}

#endif //COLLECTION_SAMPLER_H
