//
// Created by auser on 5/12/24.
//

#include <cmath>
#include "Mesh.h"
#include "OBJLoader.h"

Mesh::Mesh(): material(), triangles() {}

void Mesh::setMaterial(const Material& _material ) {
    material = _material;
}

Material Mesh::getMaterial() const {
    return material;
}

Vector3f Mesh::getSamplePoint() {
    int size = (int) triangles.size();
    if ( size == 0 ) return {};
    int ind = std::floor( rand() / (float) RAND_MAX * ( (float) size - 1 ) );
    return triangles[ind].getSamplePoint();
}

void Mesh::loadMesh(const std::string& path ) {
    OBJLoader::load( path, this );
}

void Mesh::rotate(const Vector3f& axis, float angle ) {
    Vector3f origin = getOrigin();
    move( origin * ( -1 ) );
    for ( auto& triangle: triangles ) {
        triangle.rotate( axis, angle );
    }
    move( origin );
}

void Mesh::move(const Vector3f& p ) {
    for ( auto& triangle: triangles ) {
        triangle.move( p );
    }
}

void Mesh::moveTo(const Vector3f& point ) {
    move( point - getOrigin() );
}

void Mesh::scale(float scaleValue ) {
    Vector3f oldOrigin = getOrigin();
    for ( auto& triangle: triangles ) {
        triangle.scale( scaleValue );
    }
    moveTo( oldOrigin );
}
void Mesh::scale(const Vector3f& scaleVec ) {
    Vector3f oldOrigin = getOrigin();
    for ( auto& triangle: triangles ) {
        triangle.scale( scaleVec );
    }
    moveTo( oldOrigin );
}

void Mesh::scaleTo(float scaleValue ) {
    BBox bbox = getBBox();
    Vector3f len = bbox.pMax - bbox.pMin;
    float maxLen = std::max ( std::max( len.getX(), len.getY() ), len.getZ());
    float cff = scaleValue / maxLen;
    scale( cff );
}

void Mesh::scaleTo(const Vector3f& scaleVec ) {
    BBox bbox = getBBox();
    Vector3f len = bbox.pMax - bbox.pMin;
    Vector3f cff = { scaleVec[0] / len[0], scaleVec[1] / len[1], scaleVec[2] / len[2] };
    scale( cff );
}

void Mesh::setMinPoint(const Vector3f& vec, int ind ) {
    Vector3f moveVec = vec - getBBox().pMin;
    for ( int i = 0; i < 3; ++i ) {
        if ( ind == -1 || ind == i ) continue;
        moveVec[i] = 0;
    }
    move( moveVec );
}

void Mesh::setMaxPoint(const Vector3f& vec, int ind ) {
    Vector3f moveVec = vec - getBBox().pMax;
    for ( int i = 0; i < 3; ++i ) {
        if ( ind == -1 || ind == i ) continue;
        moveVec[i] = 0;
    }
    move( moveVec );
}

Vector<Triangle> Mesh::getTriangles() {
    return triangles;
}

BBox Mesh::getBBox() const {
    Vector3f min = {__FLT_MAX__,__FLT_MAX__,__FLT_MAX__};
    Vector3f max = {__FLT_MIN__,__FLT_MIN__,__FLT_MIN__};
    for ( auto& triangle: triangles ) {
        BBox bbox = triangle.getBBox();
        if ( bbox.pMin[0] < min[0] ) min[0] = bbox.pMin[0];
        if ( bbox.pMin[1] < min[1] ) min[1] = bbox.pMin[1];
        if ( bbox.pMin[2] < min[2] ) min[2] = bbox.pMin[2];
        if ( bbox.pMax[0] > max[0] ) max[0] = bbox.pMax[0];
        if ( bbox.pMax[1] > max[1] ) max[1] = bbox.pMax[1];
        if ( bbox.pMax[2] > max[2] ) max[2] = bbox.pMax[2];
    }
    return { min, max };
}


Vector3f Mesh::getOrigin() const {
    Vector3f origin = {0,0,0};
    for ( auto& triangle: triangles ) {
        origin = origin + triangle.getOrigin();
    }
    return origin / (float) triangles.size();
}

bool Mesh::isContainPoint(const Vector3f& p ) const {
    for ( const auto& triangle: triangles ) {
        if ( triangle.isContainPoint( p ) ) return true;
    }
    return false;
}

IntersectionData Mesh::intersectsWithRay(const Ray& ray ) const {
    float min = __FLT_MAX__;
    Vector3f N = {};
    for ( const auto& triangle: triangles ) {
        float t = triangle.intersectsWithRay( ray );
        if ( t >= min ) continue;
        min = t;
    }
    return { min, N , nullptr, nullptr };
}

void Mesh::setTriangles(Vector<Triangle>& _triangles ) {
    triangles = _triangles;
    for ( auto& triangle: triangles )
        triangle.owner = this;
}
void Mesh::addTriangle(const Triangle& triangle ) {
    triangles.push_back( triangle );
    triangles[ triangles.size() - 1 ].owner = this;
}
