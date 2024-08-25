//
// Created by auser on 5/12/24.
//

#include "Sphere.h"
#include <cmath>
#include "Utils.h"
Sphere::Sphere(): radius(0), origin() {
}
Sphere::Sphere( float r, const Vector3f& pos ): radius(r), origin(pos) {
}

Sphere::Sphere( float r, const Vector3f& pos, const Material& m ): radius(r), origin(pos) {
    material = m;
}

void Sphere::rotate( const Vector3f& axis, float angle ) {
    Mat3f rotation = Mat3f::getRotationMatrix( axis, angle );
    Vector3f oldOrigin = origin;
    move( origin * ( -1 ));
    origin = rotation * origin;
    move( oldOrigin );
}

void Sphere::move( const Vector3f& p ) {
    origin = origin + p;
}

void Sphere::moveTo( const Vector3f& point ) {
    move( point - origin );
}

void Sphere::scale( float scaleValue ) {
    radius = radius * scaleValue;
}
void Sphere::scale( const Vector3f& scaleVec ) {
    if ( scaleVec[0] != scaleVec[1] || scaleVec[0] != scaleVec[2] ) return; //not sphere! nothing to do
    radius = radius * scaleVec[0];
}

void Sphere::scaleTo( float scaleValue ) {
    BBox bbox = getBBox();
    float len = bbox.pMax[0] - bbox.pMin[0];
    float cff = scaleValue / len;
    scale( cff );
}

void Sphere::scaleTo( const Vector3f& scaleVec ) {
    if ( scaleVec[0] != scaleVec[1] || scaleVec[0] != scaleVec[2] ) return; //not sphere! nothing to do
    BBox bbox = getBBox();
    float len = bbox.pMax[0] - bbox.pMin[0];
    float cff = scaleVec[0] / len;
    scale( cff );
}

void Sphere::setMaterial( const Material& mat ) {
    material = mat;
}

Vector3f Sphere::getSamplePoint() const {
    float theta = rand() / (float) RAND_MAX * M_PI;  // Угол от 0 до π
    float phi = rand() / (float) RAND_MAX * 2 * M_PI;  // Угол от 0 до 2π

    Vector3f P;
    P.x = radius * sin(theta) * cos(phi);
    P.y = radius * sin(theta) * sin(phi);
    P.z = radius * cos(theta);

    return P + origin;
}

BBox Sphere::getBBox() const {
    Vector3f r = { radius, radius, radius };
    return { origin - r, origin + r };
}

bool Sphere::isContainPoint( const Vector3f& p ) const {
    if ( getDistance( p, origin ) == radius ) return true;
    return false;
}

float Sphere::intersectsWithRay( const Ray& ray ) const {
    Vector3f D = ray.direction;
    Vector3f OC = ray.origin - origin;
    float k1 = dot( D, D );
    float k2 = 2 * dot( OC, D );
    float k3 = dot( OC, OC ) - radius * radius;

    float disc = k2 * k2 - 4 * k1 * k3;
    if ( disc < 0 ) {
        return {};
    }
    disc = sqrt( disc ) / ( 2 * k1 );
    k2 = -k2 / ( 2 * k1 );
    float t1 = k2 + disc;
    float t2 = k2 - disc;
    if ( t1 < t2 ) t2 = t1;
    return t2;
}

Material Sphere::getMaterial() const {
    return material;
}

Vector3f Sphere::getOrigin() const {
    return origin;
}

int Sphere::getIndex( const Vector3f& P, const ImageData& imageData ) const {
    Vector3f N = ( P - origin ).normalize();
    Vector3f pointOnSphere = N;
    float u = 0.5f + std::atan2( pointOnSphere.z, pointOnSphere.x) / (2 * M_PI);
    float v = 0.5f - std::asin( pointOnSphere.y ) / M_PI;
    int x = (int) (u * radius * 0.1 * imageData.width) % imageData.width;
    int y = (int) (v * radius * 0.1 * imageData.height) % imageData.height;
    return (y * imageData.width + x) * imageData.channels;
}

Vector3f Sphere::getNormal( const Vector3f& p ) const {
    Vector3f N = ( p - origin ).normalize();
    if ( !material.getTexture().normalMap.data ) return N;
    constexpr float F2_255 = 2 / 255.0f;
    int ind = getIndex( p, material.getTexture().normalMap );
    Vector3f T = { N.y, N.z, N.x };
    T = T.cross(N).normalize();
    Vector3f B = N.cross(T).normalize();
    Vector3f res = {
            (float) material.getTexture().normalMap.data[ind    ] * F2_255 - 1,
            (float) material.getTexture().normalMap.data[ind + 1] * F2_255 - 1,
            (float) material.getTexture().normalMap.data[ind + 2] * F2_255 - 1
    };
    Mat3f rot = {
            T,
            B,
            N
    };
    res = rot * res;
    return res;
}

RGB Sphere::getColor( const Vector3f& P ) const {
    if ( !material.getTexture().colorMap.data ) return material.getColor();
    int ind = getIndex( P, material.getTexture().colorMap );
    return {
            (float) material.getTexture().colorMap.data[ind    ] * 1.0f,
            (float)  material.getTexture().colorMap.data[ind + 1] * 1.0f,
            (float) material.getTexture().colorMap.data[ind + 2] * 1.0f
    };
}

RGB Sphere::getAmbient( const Vector3f& P ) const {
    if ( !material.getTexture().ambientMap.data ) return { 1, 1, 1 };
    constexpr float F1_255 = 1 / 255.0f;
    int ind = getIndex( P, material.getTexture().ambientMap );
    return {
            (float) material.getTexture().ambientMap.data[ind    ] * F1_255,
            (float) material.getTexture().ambientMap.data[ind + 1] * F1_255,
            (float) material.getTexture().ambientMap.data[ind + 2] * F1_255
    };
}

float Sphere::getRoughness( const Vector3f& P ) const {
    if ( !material.getTexture().roughnessMap.data ) return material.getRoughness();
    constexpr float F1_255 = 1 / 255.0f;
    int ind = getIndex( P, material.getTexture().roughnessMap );
    return (float) material.getTexture().roughnessMap.data[ind] * F1_255;
}