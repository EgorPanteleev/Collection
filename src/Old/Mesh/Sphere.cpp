//
// Created by auser on 5/12/24.
//

#include "Sphere.h"
#include "Mat3.h"
#include <cmath>
#include "Random.h"

Sphere::Sphere(): radius(0) {
    bbox = { origin - radius, origin + radius };
}
Sphere::Sphere( double r, const Vec3d& pos ): radius(r) {
    origin = pos;
    bbox = { origin - r, origin + r };
}

Sphere::Sphere( double r, const Vec3d& pos, const Material& m ): radius(r) {
    origin = pos;
    material = m;
    bbox = { origin - r, origin + r };
}

void Sphere::rotate( const Vec3d& axis, double angle ) {
    Mat3d rotation = Mat3d::getRotationMatrix( axis, angle );
    Vec3d oldOrigin = origin;
    move( origin * ( -1 ));
    origin = rotation * origin;
    move( oldOrigin );
}

void Sphere::move( const Vec3d& p ) {
    origin = origin + p;
}

void Sphere::moveTo( const Vec3d& point ) {
    move( point - origin );
}

void Sphere::scale( double scaleValue ) {
    radius = radius * scaleValue;
}
void Sphere::scale( const Vec3d& scaleVec ) {
    if ( scaleVec[0] != scaleVec[1] || scaleVec[0] != scaleVec[2] ) return; //not sphere! nothing to do
    radius = radius * scaleVec[0];
}

void Sphere::scaleTo( double scaleValue ) {
    BBox bbox = getBBox();
    double len = bbox.pMax[0] - bbox.pMin[0];
    double cff = scaleValue / len;
    scale( cff );
}

void Sphere::scaleTo( const Vec3d& scaleVec ) {
    if ( scaleVec[0] != scaleVec[1] || scaleVec[0] != scaleVec[2] ) return; //not sphere! nothing to do
    BBox bbox = getBBox();
    double len = bbox.pMax[0] - bbox.pMin[0];
    double cff = scaleVec[0] / len;
    scale( cff );
}

Vec3d Sphere::getSamplePoint() const {
    double theta = randomDouble() * M_PI;  // Угол от 0 до π
    double phi = randomDouble() * 2 * M_PI;  // Угол от 0 до 2π

    Vec3d P;
    P[0] = radius * sin(theta) * cos(phi);
    P[1] = radius * sin(theta) * sin(phi);
    P[2] = radius * cos(theta);

    return P + origin;
}

bool Sphere::isContainPoint( const Vec3d& p ) const {
    if ( getDistance( p, origin ) == radius ) return true;
    return false;
}

double Sphere::intersectsWithRay( const Ray& ray ) const {
    Vec3d D = ray.direction;
    Vec3d OC = ray.origin - origin;
    double k1 = dot( D, D );
    double k2 = 2 * dot( OC, D );
    double k3 = dot( OC, OC ) - radius * radius;

    double disc = k2 * k2 - 4 * k1 * k3;
    if ( disc < 0 ) {
        return {};
    }
    disc = sqrt( disc ) / ( 2 * k1 );
    k2 = -k2 / ( 2 * k1 );
    double t1 = k2 + disc;
    double t2 = k2 - disc;
    if ( t1 < t2 ) t2 = t1;
    return t2;
}

int Sphere::getIndex( const Vec3d& P, const ImageData& imageData ) const {
    Vec3d N = ( P - origin ).normalize();
    Vec3d pointOnSphere = N;
    double u = 0.5 + std::atan2( pointOnSphere[2], pointOnSphere[0]) / (2 * M_PI);
    double v = 0.5 - std::asin( pointOnSphere[1] ) / M_PI;
    int x = (int) (u * radius * 0.15 * imageData.width) % imageData.width;
    int y = (int) (v * radius * 0.15 * imageData.height) % imageData.height;
    return (y * imageData.width + x) * imageData.channels;
}

Vec3d Sphere::getNormal( const Vec3d& p ) const {
    Vec3d N = ( p - origin ).normalize();
    if ( !material.getTexture().normalMap.data ) return N;
    constexpr double F2_255 = 2 / 255.0;
    int ind = getIndex( p, material.getTexture().normalMap );
    Vec3d T = N;
    T = cross( T, N ).normalize();
    Vec3d B = cross( T, N ).normalize();
    Vec3d res = {
            material.getTexture().normalMap.data[ind    ] * F2_255 - 1,
            material.getTexture().normalMap.data[ind + 1] * F2_255 - 1,
            material.getTexture().normalMap.data[ind + 2] * F2_255 - 1
    };
    Mat3d rot = {
            T,
            B,
            N
    };
    res = rot * res;
    return res;
}