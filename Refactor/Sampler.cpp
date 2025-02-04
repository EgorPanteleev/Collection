//
// Created by auser on 8/23/24.
//

#include <algorithm>
#include "Random.h"
#include "Sampler.h"
#include "Utils.h"
#include "Interval.h"

//DEVICE Vec3d CosineWeighted::getSample( const Vec3d& N, hiprandState& state ) {
//    double u = randomDouble( 0, 1, state );
//    double v = randomDouble( 0, 1, state );
//    double r = std::sqrt(u);
//    double azimuth = v * 2 * M_PI;
//    return { r * std::cos(azimuth), r * std::sin(azimuth), std::sqrt(1 - u) };
//}

DEVICE double CosineWeighted::PDF( double Nwi ) {
    return cos( Nwi ) * M_1_PI;
}

DEVICE double CosineWeighted::PDF( const Vec3d& N, const Vec3d& wi ) {
    return PDF( dot( N, wi ) );
}

DEVICE double LambertianSampler::BRDF() {
    return pd * M_1_PI;
}

//DEVICE Vec3d LambertianSampler::getIncidentDir( const Vec3d& N, hiprandState& state ) {
//    return CosineWeighted::getSample( N, state );
//}

DEVICE double LambertianSampler::PDF( double Nwi ) {
    return CosineWeighted::PDF( Nwi );
}

DEVICE double LambertianSampler::PDF( const Vec3d& N, const Vec3d& wi ) {
    return CosineWeighted::PDF( N, wi );
}

DEVICE double OrenNayar::BRDF( const Vec3d& wi, const Vec3d& wo, double alpha ) {
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

DEVICE Vec3d OrenNayar::getIncidentDir( const Vec3d& N, hiprandState& state ) {
    return LambertianSampler::getIncidentDir( N, state ); // move to cosine namespace
}

DEVICE double OrenNayar::PDF( double Nwi ) {
    return LambertianSampler::PDF( Nwi );
}

DEVICE double OrenNayar::PDF( const Vec3d& N, const Vec3d& wi ) {
    return LambertianSampler::PDF( N, wi );
}

DEVICE double GGX::F( double dmwo ) {
    //for dielectric F0 << 0.9
    double F0 = 0.9;
    return F0 + (1.0 - F0) * pow(1.0 - dmwo, 5.0);
}

DEVICE double GGX::D( const Vec3d& m, const Vec2d& a ) {
    return 1.0 / ( M_PI * a[0] * a[1] * pow2( pow2( m[0] / a[0] ) + pow2( m[1] / a[1] ) + pow2( m[2] ) ) );
}

DEVICE double GGX::Lambda( const Vec3d& wo, const Vec2d& a ) {
    return ( -1.0 + std::sqrt( 1.0 + ( pow2 (a[0] * wo[0]) + pow2 (a[1] * wo[1]) ) / ( pow2(wo[2]) ) ) ) / 2.0;
}

DEVICE double GGX::SmithG1( const Vec3d& wo, const Vec2d& a ) {
    return 1.0 / (1.0 + Lambda(wo, a) );
}

DEVICE double GGX::DV( const Vec3d& m, const Vec3d& wo, const Vec2d& a ) {
    return SmithG1(wo, a) * dot(wo, m) * D(m, a) / wo[2];
}

DEVICE double GGX::SmithG2( const Vec3d& wi, const Vec3d& wo, const Vec2d& a ) {
    return 1.0 / (1.0 + Lambda(wo, a) + Lambda(wi, a));
}

DEVICE double GGX::BRDF( const Vec3d& wi, const Vec3d& wo, const Vec2d& a, double& PDF ) {
    Vec3d m = (wo + wi).normalize();
    PDF = DV(m, wo, a) / ( 4.0 * dot(m, wo));
    return F( dot( m, wo ) ) * D(m, a) * SmithG2(wi, wo, a) / ( 4.0 * wo[2] * wi[2]);
}

DEVICE Vec3d GGX::getNormal( const Vec3d& wo, const Vec2d& a, hiprandState& state ) {

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
