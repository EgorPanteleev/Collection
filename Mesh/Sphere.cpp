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
    BBoxData bbox = getBBox();
    float len = bbox.pMax[0] - bbox.pMin[0];
    float cff = scaleValue / len;
    scale( cff );
}

void Sphere::scaleTo( const Vector3f& scaleVec ) {
    if ( scaleVec[0] != scaleVec[1] || scaleVec[0] != scaleVec[2] ) return; //not sphere! nothing to do
    BBoxData bbox = getBBox();
    float len = bbox.pMax[0] - bbox.pMin[0];
    float cff = scaleVec[0] / len;
    scale( cff );
}

Vector3f Sphere::getSamplePoint() {
    float theta = rand() / (float) RAND_MAX * M_PI;  // Угол от 0 до π
    float phi = rand() / (float) RAND_MAX * 2 * M_PI;  // Угол от 0 до 2π

    Vector3f P;
    P.x = radius * sin(theta) * cos(phi);
    P.y = radius * sin(theta) * sin(phi);
    P.z = radius * cos(theta);

    return P + origin;
}

BBoxData Sphere::getBBox() const {
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

Vector3f Sphere::getNormal( const Vector3f& p ) const {
    Vector3f N = ( p - origin ).normalize();
    if ( !material.getTexture().normalMap.data ) return N;
    Vector3f pointOnSphere = N;
    float u = 0.5f + atan2( pointOnSphere.z, pointOnSphere.x) / (2 * M_PI);
    float v = 0.5f - asin( pointOnSphere.y ) / M_PI;

    int x = static_cast<int>(u * material.getTexture().normalMap.width) % material.getTexture().normalMap.width;
    int y = static_cast<int>(v * material.getTexture().normalMap.height) % material.getTexture().normalMap.height;
    int ind = (y * material.getTexture().normalMap.width + x) * material.getTexture().normalMap.channels;


    Vector3f T = { N.y, N.z, N.x };
    T = T.cross(N).normalize();

    Vector3f B = N.cross(T).normalize(); // Битангенс

    Vector3f res = {
            material.getTexture().normalMap.data[ind    ] / 255.0f * 2 - 1,
            material.getTexture().normalMap.data[ind + 1] / 255.0f * 2 - 1,
            material.getTexture().normalMap.data[ind + 2] / 255.0f * 2 - 1
    };


    Mat3f rot = {
            T,
            B,
            N
    };
    res = rot * res;

    return res;
}

Material Sphere::getMaterial() const {
    return material;
}

Vector3f Sphere::getOrigin() const {
    return origin;
}

RGB Sphere::getColor( const Vector3f& P ) const {
    if ( !material.getTexture().colorMap.data ) return material.getColor();
    Vector3f pointOnSphere = (P - origin).normalize();
    float u = 0.5f + atan2( pointOnSphere.z, pointOnSphere.x) / (2 * M_PI);
    float v = 0.5f - asin( pointOnSphere.y ) / M_PI;

    int x = static_cast<int>(u * material.getTexture().colorMap.width) % material.getTexture().colorMap.width;
    int y = static_cast<int>(v * material.getTexture().colorMap.height) % material.getTexture().colorMap.height;
    int ind = (y * material.getTexture().colorMap.width + x) * material.getTexture().colorMap.channels;

    return {
            material.getTexture().colorMap.data[ind    ] / 1.0f,
            material.getTexture().colorMap.data[ind + 1] / 1.0f,
            material.getTexture().colorMap.data[ind + 2] / 1.0f
    };
}

RGB Sphere::getAmbient( const Vector3f& P ) const {
    if ( !material.getTexture().ambientMap.data ) return { 1, 1, 1 };
    Vector3f pointOnSphere = (P - origin).normalize();
    float u = 0.5f + atan2( pointOnSphere.z, pointOnSphere.x) / (2 * M_PI);
    float v = 0.5f - asin( pointOnSphere.y ) / M_PI;

    int x = static_cast<int>(u * material.getTexture().ambientMap.width) % material.getTexture().ambientMap.width;
    int y = static_cast<int>(v * material.getTexture().ambientMap.height) % material.getTexture().ambientMap.height;
    int ind = (y * material.getTexture().ambientMap.width + x) * material.getTexture().ambientMap.channels;

    return {
            material.getTexture().ambientMap.data[ind    ] / 255.0f,
            material.getTexture().ambientMap.data[ind + 1] / 255.0f,
            material.getTexture().ambientMap.data[ind + 2] / 255.0f
    };
}

float Sphere::getRoughness( const Vector3f& P ) const {
    if ( !material.getTexture().roughnessMap.data ) return 0.5;
    Vector3f pointOnSphere = (P - origin).normalize();
    float u = 0.5f + atan2( pointOnSphere.z, pointOnSphere.x) / (2 * M_PI);
    float v = 0.5f - asin( pointOnSphere.y ) / M_PI;

    int x = static_cast<int>(u * material.getTexture().roughnessMap.width) % material.getTexture().roughnessMap.width;
    int y = static_cast<int>(v * material.getTexture().roughnessMap.height) % material.getTexture().roughnessMap.height;
    int ind = (y * material.getTexture().roughnessMap.width + x) * material.getTexture().roughnessMap.channels;

    return material.getTexture().roughnessMap.data[ind] / 255.0f;

}