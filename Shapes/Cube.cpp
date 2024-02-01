//
// Created by igor on 08.01.2024.
//

#include "Cube.h"

Cube::Cube(): p1({0,0,0}), p2({1,1,1}) {
    fillTriangles();
}
Cube::Cube( const Vector3f& _p1, const Vector3f& _p2): p1(_p1), p2(_p2) {
    fillTriangles();
}

void Cube::fillTriangles() {
    Vector3f f1 = p1;
    Vector3f f2 = { p2.getX(), p1.getY(), p1.getZ() };
    Vector3f f3 = { p2.getX(), p1.getY(), p2.getZ() };
    Vector3f f4 = { p1.getX(), p1.getY(), p2.getZ() };

    Vector3f b1 = { p1.getX(), p2.getY(), p1.getZ() };
    Vector3f b2 = { p2.getX(), p2.getY(), p1.getZ() };
    Vector3f b3 = p2;
    Vector3f b4 = { p1.getX(), p2.getY(), p2.getZ() };
    // down
    triangles.emplace_back( f1, f2, f3 );
    triangles.emplace_back( f1, f3, f4 );
    //up
    triangles.emplace_back( b1, b3, b2 );
    triangles.emplace_back( b1, b4, b3 );
    //left
    triangles.emplace_back( b1, f1, f4 );
    triangles.emplace_back( b1, f4, b4 );
    //right
    triangles.emplace_back( f2, b2, f3 );
    triangles.emplace_back( f3, b2, b3 );
    //front
    triangles.emplace_back( f2, f1, b1 );
    triangles.emplace_back( f2, b1, b2 );
    //back
    triangles.emplace_back( f4, f3, b4 );
    triangles.emplace_back( f3, b3, b4 );
}

void Cube::rotate( const Vector3f& axis, float angle ) {
    Vector3f origin = getOrigin();
    move( origin * ( -1 ) );
    for ( auto& triangle: triangles ) {
        triangle.rotate( axis, angle );
    }
    move( origin );
}

void Cube::move( const Vector3f& p ) {
    for ( auto& triangle: triangles ) {
        triangle.move( p );
    }
}

Vector3f Cube::getOrigin() const {
    Vector3f origin = {0,0,0};
    for ( auto& triangle: triangles ) {
        origin = origin + triangle.getOrigin();
    }
    return origin / (float) triangles.size();
}

bool Cube::isContainPoint( const Vector3f& p ) const {
    for ( const auto& triangle: triangles ) {
        if ( triangle.isContainPoint( p ) ) return true;
    }
    return false;
}

IntersectionData Cube::intersectsWithRay( const Ray& ray ) const {
    float min = std::numeric_limits<float>::max();
    Vector3f N = {};
    for ( const auto& triangle: triangles ) {
        float t = triangle.intersectsWithRay( ray );
        if ( t >= min ) continue;
        min = t;
        N = triangle.getNormal();
    }
    return { min, N };
}

Vector3f Cube::getNormal( const Vector3f& p ) const {
    return {};
}