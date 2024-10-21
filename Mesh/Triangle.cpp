#include "Triangle.h"

#include <algorithm>
#include <cmath>
#include "Mat3.h"
#include "Random.h"
//Triangle::

BBox calcBBox( Triangle* tri ) {
    double minX = std::min( std::min( tri->v1[0], tri->v2[0] ), tri->v3[0] );
    double maxX = std::max( std::max( tri->v1[0], tri->v2[0] ), tri->v3[0] );
    double minY = std::min( std::min( tri->v1[1], tri->v2[1] ), tri->v3[1] );
    double maxY = std::max( std::max( tri->v1[1], tri->v2[1] ), tri->v3[1] );
    double minZ = std::min( std::min( tri->v1[2], tri->v2[2] ), tri->v3[2] );
    double maxZ = std::max( std::max( tri->v1[2], tri->v2[2] ), tri->v3[2] );
    return { Vec3d( minX, minY, minZ ), Vec3d( maxX, maxY, maxZ ) };
}


Triangle::Triangle(): v1(), v2(), v3() {
    edge1 = v2 - v1;
    edge2 = v3 - v1;
    N = cross( edge1, edge2 ).normalize();
    origin = (v1 + v2 + v3) / 3;
    double xMax = std::max( std::max( v1[0], v2[0] ), v3[0] );
    double yMax = std::max( std::max( v1[1], v2[1] ), v3[1] );
    double xMin = std::min( std::min( v1[0], v2[0] ), v3[0] );
    double yMin = std::min( std::min( v1[1], v2[1] ), v3[1] );
    v1Tex = { (v1[0] - xMin) / (xMax - xMin), (v1[1] - yMin) / (yMax - yMin) };
    v2Tex = { (v2[0] - xMin) / (xMax - xMin), (v2[1] - yMin) / (yMax - yMin) };
    v3Tex = { (v3[0] - xMin) / (xMax - xMin), (v3[1] - yMin) / (yMax - yMin) };
    bbox = calcBBox( this );
}

Triangle::Triangle( const Vec3d& v1, const Vec3d& v2, const Vec3d& v3 ): v1( v1 ), v2( v2 ), v3( v3 ) {
    edge1 = v2 - v1;
    edge2 = v3 - v1;
    N = cross( edge1, edge2 ).normalize();
    origin = (v1 + v2 + v3) / 3;
    Vec3d targetNormal(0, 0, -1);

    Vec3d axis = cross( targetNormal, N );
    double angle = acos( dot( targetNormal, N ) ) * 180 * M_1_PI;

    Mat3d rotationMatrix = Mat3d::getRotationMatrix( axis, angle );

    Vec3d rv1 = v1 * rotationMatrix;
    Vec3d rv2 = v2 * rotationMatrix;
    Vec3d rv3 = v3 * rotationMatrix;
    double xMax = std::max( std::max( rv1[0], rv2[0] ), rv3[0] );
    double yMax = std::max( std::max( rv1[1], rv2[1] ), rv3[1] );

    double xMin = std::min( std::min( rv1[0], rv2[0] ), rv3[0] );
    double yMin = std::min( std::min( rv1[1], rv2[1] ), rv3[1] );
    //TODO remove hardcode
    double xTex = ( xMax - xMin ) / 150.0;
    double yTex = ( yMax - yMin ) / 93.75;

//    xTex = std::min( xTex, 1.0 );
//    yTex = std::min( yTex, 1.0 );

    v1Tex = { rv1[0] == xMax ? xTex : 0.0, rv1[1] == yMax ? yTex : 0.0 };
    v2Tex = { rv2[0] == xMax ? xTex : 0.0, rv2[1] == yMax ? yTex : 0.0 };
    v3Tex = { rv3[0] == xMax ? xTex : 0.0, rv3[1] == yMax ? yTex : 0.0 };
    bbox = calcBBox( this );
}

void Triangle::rotate( const Vec3d& axis, double angle ) {
    Mat3d rotation = Mat3d::getRotationMatrix( axis, angle );
    v1 = rotation * v1;
    v2 = rotation * v2;
    v3 = rotation * v3;
    edge1 = v2 - v1;
    edge2 = v3 - v1;
    N = cross( edge1, edge2 ).normalize();
    origin = (v1 + v2 + v3) / 3;
    bbox = calcBBox( this );
}

void Triangle::move( const Vec3d& p ) {
    v1 = v1 + p;
    v2 = v2 + p;
    v3 = v3 + p;
    origin = (v1 + v2 + v3) / 3;
    bbox = calcBBox( this );
}

void Triangle::moveTo( const Vec3d& point ) {
    move( point - getOrigin() );
}

void Triangle::scale( double scaleValue ) {
    v1 = v1 * scaleValue;
    v2 = v2 * scaleValue;
    v3 = v3 * scaleValue;
    edge1 = v2 - v1;
    edge2 = v3 - v1;
    N = cross( edge1, edge2 ).normalize();
    origin = (v1 + v2 + v3) / 3;
    bbox = calcBBox( this );
}
void Triangle::scale( const Vec3d& scaleVec ) {
    v1 = { v1[0] * scaleVec[0], v1[1] * scaleVec[1], v1[2] * scaleVec[2] };
    v2 = { v2[0] * scaleVec[0], v2[1] * scaleVec[1], v2[2] * scaleVec[2] };
    v3 = { v3[0] * scaleVec[0], v3[1] * scaleVec[1], v3[2] * scaleVec[2] };
    edge1 = v2 - v1;
    edge2 = v3 - v1;
    N = cross( edge1, edge2 ).normalize();
    origin = (v1 + v2 + v3) / 3;
    bbox = calcBBox( this );
}

void Triangle::scaleTo( double scaleValue ) {
    BBox bbox = getBBox();
    Vec3d len = bbox.pMax - bbox.pMin;
    Vec3d cff = { scaleValue / len[0], scaleValue / len[1], scaleValue / len[2] };
    scale( cff );
}
void Triangle::scaleTo( const Vec3d& scaleVec ) {
    BBox bbox = getBBox();
    Vec3d len = bbox.pMax - bbox.pMin;
    Vec3d cff = { scaleVec[0] / len[0], scaleVec[1] / len[1], scaleVec[2] / len[2] };
    scale( cff );
}

Vec3d Triangle::getSamplePoint() const {
        double u = randomDouble();
        double v = randomDouble();
        if (u + v > 1.0) {
            u = 1.0 - u;
            v = 1.0 - v;
        }
        Vec3d P = v1 + edge1 * u + edge2 * v;
        return P;
}

bool Triangle::isContainPoint( const Vec3d& p ) const {
    double detT = (v2[1] - v3[1]) * (v1[0] - v3[0]) + (v3[0] - v2[0]) * (v1[1] - v3[1]);
    double alpha = ((v2[1] - v3[1]) * (p[0] - v3[0]) + (v3[0] - v2[0]) * (p[1] - v3[1])) / detT;
    double beta = ((v3[1] - v1[1]) * (p[0] - v3[0]) + (v1[0] - v3[0]) * (p[1] - v3[1])) / detT;
    double gamma = 1.0 - alpha - beta;

    // Check if the point is inside the triangle
    return ( alpha >= 0.0 && alpha <= 1.0 &&
           beta >= 0.0 && beta <= 1.0 &&
           gamma >= 0.0 && gamma <= 1.0 &&
           p[2] >=std::min( std::min(v1[2], v2[2]), v3[2] ) &&
           p[2] <= std::max(std::max(v1[2], v2[2]), v3[2] ) );
}

double Triangle::intersectsWithRay( const Ray& ray ) const {
    Vec3d h = cross( ray.direction, edge2 );
    double a = dot(edge1, h);

    if ( a < __FLT_EPSILON__ ) return __FLT_MAX__;

    double f = 1.0 / a;
    Vec3d s = ray.origin - v1;
    double u = f * dot(s, h);

    if ( u < 0.0 || u > 1.0 ) return __FLT_MAX__;

    Vec3d q = cross( s, edge1 );
    double v = f * dot(ray.direction, q);

    if  ( v < 0.0 || u + v > 1.0 ) return __FLT_MAX__;

    double t = f * dot(edge2, q);

    if ( t < __FLT_EPSILON__ ) return __FLT_MAX__;

    return t;
}

int Triangle::getIndex( const Vec3d& P, const ImageData& imageData ) const {
    Vec3d edge3 = P - v1;
    double d00 = dot(edge1, edge1);
    double d01 = dot(edge1, edge2);
    double d11 = dot(edge2, edge2);
    double d20 = dot(edge3, edge1);
    double d21 = dot(edge3, edge2);
    double denom = d00 * d11 - d01 * d01;
    double v = (d11 * d20 - d01 * d21) / denom;
    double w = (d00 * d21 - d01 * d20) / denom;
    double u = 1.0 - v - w;

    double tx = u * v1Tex[0] + v * v2Tex[0] + w * v3Tex[0];
    double ty = u * v1Tex[1] + v * v2Tex[1] + w * v3Tex[1];

    int texX = (int) (tx * imageData.width) % imageData.width;
    int texY = (int) (ty * imageData.height) % imageData.height;

    return (texY * imageData.width + texX) * imageData.channels;
}

Vec3d Triangle::getNormal( const Vec3d& P ) const {
    if ( !material.getTexture().normalMap.data ) return N;
    constexpr double F2_255 = 2 / 255.0;
    int ind = getIndex( P, material.getTexture().normalMap );
    Vec3d res = {
            material.getTexture().normalMap.data[ind    ] * F2_255 - 1,
            material.getTexture().normalMap.data[ind + 1] * F2_255 - 1,
            material.getTexture().normalMap.data[ind + 2] * F2_255 - 1
    };

    Vec3d up = (std::abs(N[2]) < 0.999f) ? Vec3d{0.0, 0.0, 1.0} : Vec3d{1.0, 0.0, 0.0};
    Vec3d tangent = cross( up, N );
    Vec3d bitangent = cross( N, tangent );
    Mat3d rot = { tangent.normalize(), bitangent.normalize(), N };
    res = rot * res;
    return res.normalize();
}