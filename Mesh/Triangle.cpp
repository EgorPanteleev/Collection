#include "Triangle.h"
#include "Utils.h"
#include <algorithm>
#include <cmath>
//Triangle::

Triangle::Triangle(): v1(), v2(), v3() {
    edge1 = v2 - v1;
    edge2 = v3 - v1;
    N = edge1.cross( edge2 ).normalize();
    origin = (v1 + v2 + v3) / 3;
    float xMax = std::max( std::max( v1.x, v2.x ), v3.x );
    float yMax = std::max( std::max( v1.y, v2.y ), v3.y );
    float xMin = std::min( std::min( v1.x, v2.x ), v3.x );
    float yMin = std::min( std::min( v1.y, v2.y ), v3.y );
    v1Tex = { (v1.x - xMin) / (xMax - xMin), (v1.y - yMin) / (yMax - yMin) };
    v2Tex = { (v2.x - xMin) / (xMax - xMin), (v2.y - yMin) / (yMax - yMin) };
    v3Tex = { (v3.x - xMin) / (xMax - xMin), (v3.y - yMin) / (yMax - yMin) };
}

Triangle::Triangle( const Vector3f& v1, const Vector3f& v2, const Vector3f& v3 ): v1( v1 ), v2( v2 ), v3( v3 ) {
    edge1 = v2 - v1;
    edge2 = v3 - v1;
    N = edge1.cross( edge2 ).normalize();
    origin = (v1 + v2 + v3) / 3;
    Vector3f targetNormal(0, 0, -1);

    Vector3f axis = targetNormal.cross(N);
    float angle = acos( dot( targetNormal, N ) ) * 180 * M_1_PI;

    Mat3f rotationMatrix = Mat3f::getRotationMatrix( axis, angle );

    Vector3f rv1 = v1 * rotationMatrix;
    Vector3f rv2 = v2 * rotationMatrix;
    Vector3f rv3 = v3 * rotationMatrix;
    float xMax = std::max( std::max( rv1.x, rv2.x ), rv3.x );
    float yMax = std::max( std::max( rv1.y, rv2.y ), rv3.y );

    float xMin = std::min( std::min( rv1.x, rv2.x ), rv3.x );
    float yMin = std::min( std::min( rv1.y, rv2.y ), rv3.y );
    //TODO remove hardcode
    float xTex = ( xMax - xMin ) / 150.0f;
    float yTex = ( yMax - yMin ) / 93.75f;

//    xTex = std::min( xTex, 1.0f );
//    yTex = std::min( yTex, 1.0f );

    v1Tex = { rv1.x == xMax ? xTex : 0.0f, rv1.y == yMax ? yTex : 0.0f };
    v2Tex = { rv2.x == xMax ? xTex : 0.0f, rv2.y == yMax ? yTex : 0.0f };
    v3Tex = { rv3.x == xMax ? xTex : 0.0f, rv3.y == yMax ? yTex : 0.0f };
}

void Triangle::rotate( const Vector3f& axis, float angle ) {
    Mat3f rotation = Mat3f::getRotationMatrix( axis, angle );
    v1 = rotation * v1;
    v2 = rotation * v2;
    v3 = rotation * v3;
    edge1 = v2 - v1;
    edge2 = v3 - v1;
    N = edge1.cross( edge2 ).normalize();
    origin = (v1 + v2 + v3) / 3;
}

void Triangle::move( const Vector3f& p ) {
    v1 = v1 + p;
    v2 = v2 + p;
    v3 = v3 + p;
    origin = (v1 + v2 + v3) / 3;
}

void Triangle::moveTo( const Vector3f& point ) {
    move( point - getOrigin() );
}

void Triangle::scale( float scaleValue ) {
    v1 = v1 * scaleValue;
    v2 = v2 * scaleValue;
    v3 = v3 * scaleValue;
    edge1 = v2 - v1;
    edge2 = v3 - v1;
    N = edge1.cross( edge2 ).normalize();
    origin = (v1 + v2 + v3) / 3;
}
void Triangle::scale( const Vector3f& scaleVec ) {
    v1 = { v1[0] * scaleVec[0], v1[1] * scaleVec[1], v1[2] * scaleVec[2] };
    v2 = { v2[0] * scaleVec[0], v2[1] * scaleVec[1], v2[2] * scaleVec[2] };
    v3 = { v3[0] * scaleVec[0], v3[1] * scaleVec[1], v3[2] * scaleVec[2] };
    edge1 = v2 - v1;
    edge2 = v3 - v1;
    N = edge1.cross( edge2 ).normalize();
    origin = (v1 + v2 + v3) / 3;
}

void Triangle::scaleTo( float scaleValue ) {
    BBox bbox = getBBox();
    Vector3f len = bbox.pMax - bbox.pMin;
    Vector3f cff = { scaleValue / len[0], scaleValue / len[1], scaleValue / len[2] };
    scale( cff );
}
void Triangle::scaleTo( const Vector3f& scaleVec ) {
    BBox bbox = getBBox();
    Vector3f len = bbox.pMax - bbox.pMin;
    Vector3f cff = { scaleVec[0] / len[0], scaleVec[1] / len[1], scaleVec[2] / len[2] };
    scale( cff );
}

void Triangle::setMaterial( const Material& mat ) {
    material = mat;
}

Vector3f Triangle::getSamplePoint() const {
        float u = randomFloat();
        float v = randomFloat();
        if (u + v > 1.0f) {
            u = 1.0f - u;
            v = 1.0f - v;
        }
        Vector3f P = v1 + edge1 * u + edge2 * v;
        return P;
}

BBox Triangle::getBBox() const {
    float minX = std::min( std::min( v1[0], v2[0] ), v3[0] );
    float maxX = std::max( std::max( v1[0], v2[0] ), v3[0] );
    float minY = std::min( std::min( v1[1], v2[1] ), v3[1] );
    float maxY = std::max( std::max( v1[1], v2[1] ), v3[1] );
    float minZ = std::min( std::min( v1[2], v2[2] ), v3[2] );
    float maxZ = std::max( std::max( v1[2], v2[2] ), v3[2] );
    return { Vector3f( minX, minY, minZ ), Vector3f( maxX, maxY, maxZ ) };
}

Vector3f Triangle::getOrigin() const {
    return origin;
}

bool Triangle::isContainPoint( const Vector3f& p ) const {
    float detT = (v2.getY() - v3.getY()) * (v1.getX() - v3.getX()) + (v3.getX() - v2.getX()) * (v1.getY() - v3.getY());
    float alpha = ((v2.getY() - v3.getY()) * (p.getX() - v3.getX()) + (v3.getX() - v2.getX()) * (p.getY() - v3.getY())) / detT;
    float beta = ((v3.getY() - v1.getY()) * (p.getX() - v3.getX()) + (v1.getX() - v3.getX()) * (p.getY() - v3.getY())) / detT;
    float gamma = 1.0f - alpha - beta;

    // Check if the point is inside the triangle
    return ( alpha >= 0.0f && alpha <= 1.0f &&
           beta >= 0.0f && beta <= 1.0f &&
           gamma >= 0.0f && gamma <= 1.0f &&
           p.getZ() >=std::min( std::min(v1.getZ(), v2.getZ()), v3.getZ() ) &&
           p.getZ() <= std::max(std::max(v1.getZ(), v2.getZ()), v3.getZ() ) );
}

float Triangle::intersectsWithRay( const Ray& ray ) const {
    Vector3f h = ray.direction.cross( edge2 );
    float a = dot(edge1, h);

    if ( a < __FLT_EPSILON__ ) return __FLT_MAX__;

    float f = 1.0f / a;
    Vector3f s = ray.origin - v1;
    float u = f * dot(s, h);

    if ( u < 0.0f || u > 1.0f ) return __FLT_MAX__;

    Vector3f q = s.cross( edge1 );
    float v = f * dot(ray.direction, q);

    if  ( v < 0.0f || u + v > 1.0f ) return __FLT_MAX__;

    float t = f * dot(edge2, q);

    if ( t < __FLT_EPSILON__ ) return __FLT_MAX__;

    return t;
}

int Triangle::getIndex( const Vector3f& P, const ImageData& imageData ) const {
    Vector3f edge3 = P - v1;
    float d00 = dot(edge1, edge1);
    float d01 = dot(edge1, edge2);
    float d11 = dot(edge2, edge2);
    float d20 = dot(edge3, edge1);
    float d21 = dot(edge3, edge2);
    float denom = d00 * d11 - d01 * d01;
    float v = (d11 * d20 - d01 * d21) / denom;
    float w = (d00 * d21 - d01 * d20) / denom;
    float u = 1.0f - v - w;

    float tx = u * v1Tex.getX() + v * v2Tex.getX() + w * v3Tex.getX();
    float ty = u * v1Tex.getY() + v * v2Tex.getY() + w * v3Tex.getY();

    int texX = (int) (tx * imageData.width) % imageData.width;
    int texY = (int) (ty * imageData.height) % imageData.height;

    return (texY * imageData.width + texX) * imageData.channels;
}

Vector3f Triangle::getNormal( const Vector3f& P ) const {
    if ( !material.getTexture().normalMap.data ) return N;
    constexpr float F2_255 = 2 / 255.0f;
    int ind = getIndex( P, material.getTexture().normalMap );
    Vector3f res = {
            (float) material.getTexture().normalMap.data[ind    ] * F2_255 - 1,
            (float) material.getTexture().normalMap.data[ind + 1] * F2_255 - 1,
            (float) material.getTexture().normalMap.data[ind + 2] * F2_255 - 1
    };

    Vector3f up = (std::abs(N.z) < 0.999f) ? Vector3f{0.0f, 0.0f, 1.0f} : Vector3f{1.0f, 0.0f, 0.0f};
    Vector3f tangent = up.cross(N);
    Vector3f bitangent = N.cross(tangent);
    Mat3f rot = { tangent.normalize(), bitangent.normalize(), N };
    res = rot * res;
    return res.normalize();
}


RGB Triangle::getColor( const Vector3f& P ) const {
    if ( !material.getTexture().colorMap.data ) return material.getColor();

    int ind = getIndex( P, material.getTexture().colorMap );
    return {
            (float) material.getTexture().colorMap.data[ind    ] * 1.0f,
            (float) material.getTexture().colorMap.data[ind + 1] * 1.0f,
            (float) material.getTexture().colorMap.data[ind + 2] * 1.0f
    };
}

RGB Triangle::getAmbient( const Vector3f& P ) const {
    if ( !material.getTexture().ambientMap.data ) return { 1, 1, 1 };
    constexpr float F1_255 = 1 / 255.0f;
    int ind = getIndex( P, material.getTexture().ambientMap );
    return {
            (float) material.getTexture().ambientMap.data[ind    ] * F1_255,
            (float) material.getTexture().ambientMap.data[ind + 1] * F1_255,
            (float) material.getTexture().ambientMap.data[ind + 2] * F1_255
    };
}

float Triangle::getRoughness( const Vector3f& P ) const {
    if ( !material.getTexture().roughnessMap.data ) return material.getRoughness();
    constexpr float F1_255 = 1 / 255.0f;
    int ind = getIndex( P, material.getTexture().roughnessMap );

    return (float) material.getTexture().roughnessMap.data[ind] * F1_255;
}
float Triangle::getMetalness( const Vector3f& P ) const {
    if ( !material.getTexture().metalnessMap.data ) return material.getMetalness();
    constexpr float F1_255 = 1 / 255.0f;
    int ind = getIndex( P, material.getTexture().metalnessMap );

    return (float) material.getTexture().metalnessMap.data[ind] * F1_255;
}

Material Triangle::getMaterial() const {
    return material;
}