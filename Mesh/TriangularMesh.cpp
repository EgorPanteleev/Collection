#include "TriangularMesh.h"
#include "OBJLoader.h"
//TriangularMesh::TriangularMesh( const std::string& path ) {
//    OBJLoader::load( path, this );
//}
//
//TriangularMesh::TriangularMesh( std::vector<Triangle> _triangles ): triangles( _triangles ) {
//}

void TriangularMesh::loadMesh( const std::string& path ) {
    OBJLoader::load( path, this );
}

void TriangularMesh::rotate( const Vector3f& axis, float angle ) {
    Vector3f origin = getOrigin();
    move( origin * ( -1 ) );
    for ( auto& triangle: triangles ) {
        triangle.rotate( axis, angle );
    }
    move( origin );
}

void TriangularMesh::move( const Vector3f& p ) {
    for ( auto& triangle: triangles ) {
        triangle.move( p );
    }
}

void TriangularMesh::moveTo( const Vector3f& point ) {
    move( point - getOrigin() );
}

void TriangularMesh::scale( float scaleValue ) {
    Vector3f oldOrigin = getOrigin();
    for ( auto& triangle: triangles ) {
        triangle.scale( scaleValue );
    }
    moveTo( oldOrigin );
}
void TriangularMesh::scale( const Vector3f& scaleVec ) {
    Vector3f oldOrigin = getOrigin();
    for ( auto& triangle: triangles ) {
        triangle.scale( scaleVec );
    }
    moveTo( oldOrigin );
}
//  OLD
//BBoxData bbox = getBBox();
//Vector3f len = bbox.pMax - bbox.pMin;
//Vector3f cff = { scaleValue / len[0], scaleValue / len[1], scaleValue / len[2] };
//scale( cff );
void TriangularMesh::scaleTo( float scaleValue ) {
    BBoxData bbox = getBBox();
    Vector3f len = bbox.pMax - bbox.pMin;
    float maxLen = std::max ( std::max( len.getX(), len.getY() ), len.getZ());
    float cff = scaleValue / maxLen;
    scale( cff );
}

void TriangularMesh::scaleTo( const Vector3f& scaleVec ) {
    BBoxData bbox = getBBox();
    Vector3f len = bbox.pMax - bbox.pMin;
    Vector3f cff = { scaleVec[0] / len[0], scaleVec[1] / len[1], scaleVec[2] / len[2] };
    scale( cff );
}

void TriangularMesh::setMinPoint( const Vector3f& vec, int ind ) {
    Vector3f moveVec = vec - getBBox().pMin;
    for ( int i = 0; i < 3; ++i ) {
        if ( ind == -1 || ind == i ) continue;
        moveVec[i] = 0;
    }
    move( moveVec );
}

void TriangularMesh::setMaxPoint( const Vector3f& vec, int ind ) {
    Vector3f moveVec = vec - getBBox().pMax;
    for ( int i = 0; i < 3; ++i ) {
        if ( ind == -1 || ind == i ) continue;
        moveVec[i] = 0;
    }
    move( moveVec );
}

std::vector<Triangle> TriangularMesh::getTriangles() {
    return triangles;
}

BBoxData TriangularMesh::getBBox() const {
    static float MAX = std::numeric_limits<float>::max();
    static float MIN = std::numeric_limits<float>::min();
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


Vector3f TriangularMesh::getOrigin() const {
    Vector3f origin = {0,0,0};
    for ( auto& triangle: triangles ) {
        origin = origin + triangle.getOrigin();
    }
    return origin / (float) triangles.size();
}

bool TriangularMesh::isContainPoint( const Vector3f& p ) const {
    for ( const auto& triangle: triangles ) {
        if ( triangle.isContainPoint( p ) ) return true;
    }
    return false;
}

IntersectionData TriangularMesh::intersectsWithRay( const Ray& ray ) const {
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

Vector3f TriangularMesh::getNormal( const Vector3f& p ) const {
    return {};
}

void TriangularMesh::setTriangles( std::vector<Triangle>& _triangles ) {
    triangles = _triangles;
    for ( auto& triangle: triangles )
        triangle.owner = this;
}
void TriangularMesh::addTriangle( const Triangle& triangle ) {
    triangles.push_back( triangle );
    triangles[ triangles.size() - 1 ].owner = this;
}
