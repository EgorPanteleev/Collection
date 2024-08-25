//
// Created by auser on 8/23/24.
//

#include "Sampler.h"

Mat3f getTBN( const Vector3f& wo, const Vector3f& N ) {
    Vector3f TBN0 = ( wo - N * dot( N, wo ) ).normalize();
    return { TBN0           ,
             N.cross( TBN0 ),
             N              };
}

Vector3f CosineWeighted::getSample( const Vector3f& N ) {
    const float u = randomFloat();
    const float v = randomFloat();
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

float CosineWeighted::PDF( float Nwi ) {
    return cos( Nwi ) * M_1_PI;
}

float CosineWeighted::PDF( const Vector3f& N, const Vector3f& wi ) {
    return PDF( dot( N, wi ) );
}

float Lambertian::BRDF() {
    return pd * M_1_PI;
}

Vector3f Lambertian::getIncidentDir( const Vector3f& N ) {
    return CosineWeighted::getSample( N );
}

float Lambertian::PDF( float Nwi ) {
    return CosineWeighted::PDF( Nwi );
}

float Lambertian::PDF( const Vector3f& N, const Vector3f& wi ) {
    return CosineWeighted::PDF( N, wi );
}

float OrenNayar::BRDF( const Vector3f& N, const Vector3f& wi, const Vector3f& wo, float alpha ) {
    float A = 1 - ( alpha / ( 2 * ( alpha + 0.33 ) ) ); // A and B precomputed
    float B = ( 0.45f * alpha ) / ( alpha + 0.09f );
    float theta_i = std::acos(dot(N, wi));
    float theta_o = std::acos(dot(N, wo));

    float angleAlpha = std::max(theta_i, theta_o);
    float angleBeta = std::min(theta_i, theta_o);

    float phi_i = std::atan2(wi.y, wi.x);
    float phi_o = std::atan2(wo.y, wo.x);

    float cosDeltaPhi = std::max( 0.0f, (float) cos( phi_i - phi_o ) );

    return Lambertian::BRDF() * (float) ( A + B * cosDeltaPhi * std::sin( angleAlpha ) * std::tan( angleBeta ) );
}

Vector3f OrenNayar::getIncidentDir( const Vector3f& N ) {
    return Lambertian::getIncidentDir( N ); // move to cosine namespace
}

float OrenNayar::PDF( float Nwi ) {
    return Lambertian::PDF( Nwi );
}

float OrenNayar::PDF( const Vector3f& N, const Vector3f& wi ) {
    return Lambertian::PDF( N, wi );
}

float GGX::D( const Vector3f& m, const Vector2f& a ) {
    return 1.0f / (float) ( M_PI * a.x * a.y * pow2( pow2( m.x / a.x ) + pow2( m.y / a.y ) + pow2( m.z ) ) );
}

float GGX::Lambda( const Vector3f& wo, const Vector2f& a ) {
    return ( -1.0f + std::sqrt( 1.0f + ( pow2 (a.x * wo.x) + pow2 (a.y * wo.y) ) / ( pow2(wo.z) ) ) ) / 2.0f;
}

float GGX::SmithG1( const Vector3f& wo, const Vector2f& a ) {
    return 1.0f / (1.0f + Lambda(wo, a) );
}

float GGX::DV( const Vector3f& m, const Vector3f& wo, const Vector2f& a ) {
    return SmithG1(wo, a) * dot(wo, m) * D(m, a) / wo.z;
}

float GGX::SmithG2( const Vector3f& wi, const Vector3f& wo, const Vector2f& a ) {
    return 1.0f / (1.0f + Lambda(wo, a) + Lambda(wi, a));
}

float GGX::BRDF( const Vector3f& wi, const Vector3f& wo, const Vector2f& a, float& PDF ) {
    Vector3f m = (wo + wi).normalize();
    PDF = DV(m, wo, a) / ( 4.0f * dot(m, wo));
    return D(m, a) * SmithG2(wi, wo, a) / ( 4.0f * wo.z * wi.z);
}

Vector3f GGX::getNormal( const Vector3f& wo, const Vector2f& a ) {

    float u = randomFloat();
    float v = randomFloat();
    Vector3f Vh = (Vector3f(a.x * wo.x, a.y * wo.y, wo.z)).normalize();

    // orthonormal basis (with special case if cross product is zero)
    float len2 = pow2( Vh.x ) + pow2( Vh.y );
    Vector3f T1 = len2 > 0.0 ? Vector3f( -Vh.y, Vh.x, 0.0 ) * 1.0f / std::sqrt(len2) : Vector3f(1.0f, 0.0f, 0.0f);
    Vector3f T2 = Vh.cross( T1 );

    // parameterization of the projected area
    float r = std::sqrt( u );
    float phi = v * 2 * (float) M_PI;
    float t1 = r * std::cos(phi);
    float t2 = r * std::sin(phi);
    float s = 0.5f * ( 1.0f + Vh.z );
    t2 = ( 1.0f - s ) * (float) std::sqrt(1.0 - pow2( t1 ) ) + s * t2;

    // reprojection onto hemisphere
    Vector3f Nh = t1 * T1 + t2 * T2 + (float) std::sqrt(std::max( 0.0f, 1.0f - pow2(t1) - pow2(t2))) * Vh;

    // transforming the normal back to the ellipsoid configuration
    return ( Vector3f(a.x * Nh.x, a.y * Nh.y, std::max(0.0f, Nh.z)) ).normalize();
}
