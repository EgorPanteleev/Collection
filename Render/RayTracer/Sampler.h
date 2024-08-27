//
// Created by auser on 8/23/24.
//

#ifndef COLLECTION_SAMPLER_H
#define COLLECTION_SAMPLER_H
#include "Vector3f.h"
#include "Utils.h"
#include <cmath>

#define pd 1.0f // between 0 and 1, how much diffuse light be diffuse

Mat3f getTBN( const Vector3f& wo, const Vector3f& N );

namespace CosineWeighted {
    Vector3f getSample( const Vector3f& N );

    float PDF( float Nwi );

    float PDF( const Vector3f& N, const Vector3f& wi );
}

namespace Lambertian {
    float BRDF();

    Vector3f getIncidentDir( const Vector3f& N );

    float PDF( float Nwi );

    float PDF( const Vector3f& N, const Vector3f& wi );
}

//we can avoid trigonometric functions for better perfomance ( but not now )

namespace OrenNayar {
    float BRDF( const Vector3f& wi, const Vector3f& wo, float alpha );

    Vector3f getIncidentDir( const Vector3f& N );

    float PDF( float Nwi );

    float PDF( const Vector3f& N, const Vector3f& wi );
}

namespace GGX {
    float D( const Vector3f& m, const Vector2f& a );

    float F( float dmwo );

    float Lambda( const Vector3f& wo, const Vector2f& a );

    float SmithG1( const Vector3f& wo, const Vector2f& a );

    float DV( const Vector3f& m, const Vector3f& wo, const Vector2f& a );

    float SmithG2( const Vector3f& wi, const Vector3f& wo, const Vector2f& a );

    float BRDF( const Vector3f& wi, const Vector3f& wo, const Vector2f& a, float& PDF );

    Vector3f getNormal( const Vector3f& wo, const Vector2f& a );
}

#endif //COLLECTION_SAMPLER_H
