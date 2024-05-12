//
// Created by igor on 08.01.2024.
//

#include "Cube.h"
#include "IntersectionData.h"
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

void Cube::moveTo( const Vector3f& point ) {
    move( point - getOrigin() );
}

void Cube::scale( float scaleValue ) {
    Vector3f oldOrigin = getOrigin();
    for ( auto& triangle: triangles ) {
        triangle.scale( scaleValue );
    }
    moveTo( oldOrigin );
}
void Cube::scale( const Vector3f& scaleVec ) {
    Vector3f oldOrigin = getOrigin();
    for ( auto& triangle: triangles ) {
        triangle.scale( scaleVec );
    }
    moveTo( oldOrigin );
}

void Cube::scaleTo( float scaleValue ) {
    BBoxData bbox = getBBox();
    Vector3f len = bbox.pMax - bbox.pMin;
    Vector3f cff = { scaleValue / len[0], scaleValue / len[1], scaleValue / len[2] };
    scale( cff );
}

void Cube::scaleTo( const Vector3f& scaleVec ) {
    BBoxData bbox = getBBox();
    Vector3f len = bbox.pMax - bbox.pMin;
    Vector3f cff = { scaleVec[0] / len[0], scaleVec[1] / len[1], scaleVec[2] / len[2] };
    scale( cff );

}
std::vector<Triangle> Cube::getTriangles() { return triangles; }

BBoxData Cube::getBBox() const {
    static float MAX = std::numeric_limits<float>::max();
    static float MIN = -std::numeric_limits<float>::max();
    Vector3f min = {MAX,MAX,MAX};
    Vector3f max = {MIN,MIN,MIN};
    for ( auto& triangle: triangles ) {
        BBoxData bbox = triangle.getBBox();
        if ( bbox.pMin[0] < min[0] ) min[0] = bbox.pMin[0];
        if ( bbox.pMin[1] < min[1] ) min[1] = bbox.pMin[1];
        if ( bbox.pMin[2] < min[2] ) min[2] = bbox.pMin[2];
        if ( bbox.pMax[0] > max[0] ) max[0] = bbox.pMax[0];
        if ( bbox.pMax[1] > max[1] ) max[1] = bbox.pMax[1];
        if ( bbox.pMax[2] > max[2] ) max[2] = bbox.pMax[2];
    }
    return { min, max };
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
    return { min, N , nullptr};
}

Vector3f Cube::getNormal( const Vector3f& p ) const {
    return {};
}