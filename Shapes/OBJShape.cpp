#include "OBJShape.h"
#include "OBJLoader.h"
OBJShape::OBJShape( const std::string& path ) {
    OBJLoader::load( path, this );
}

void OBJShape::rotate( const Vector3f& axis, float angle ) {
    Vector3f origin = getOrigin();
    move( origin * ( -1 ) );
    for ( auto& triangle: triangles ) {
        triangle.rotate( axis, angle );
    }
    move( origin );
}

void OBJShape::move( const Vector3f& p ) {
    for ( auto& triangle: triangles ) {
        triangle.move( p );
    }
}

void OBJShape::moveTo( const Vector3f& point ) {
    move( point - getOrigin() );
}

void OBJShape::scale( float scaleValue ) {
    Vector3f oldOrigin = getOrigin();
    for ( auto& triangle: triangles ) {
        triangle.scale( scaleValue );
    }
    moveTo( oldOrigin );
}
void OBJShape::scale( const Vector3f& scaleVec ) {
    Vector3f oldOrigin = getOrigin();
    for ( auto& triangle: triangles ) {
        triangle.scale( scaleVec );
    }
    moveTo( oldOrigin );
}

void OBJShape::scaleTo( float scaleValue ) {
    BBoxData bbox = getBBox();
    Vector3f len = bbox.pMax - bbox.pMin;
    Vector3f cff = { scaleValue / len[0], scaleValue / len[1], scaleValue / len[2] };
    scale( cff );
}

void OBJShape::scaleTo( const Vector3f& scaleVec ) {
    BBoxData bbox = getBBox();
    Vector3f len = bbox.pMax - bbox.pMin;
    Vector3f cff = { scaleVec[0] / len[0], scaleVec[1] / len[1], scaleVec[2] / len[2] };
    scale( cff );
}

std::vector<Triangle> OBJShape::getTriangles() { return triangles; }


BBoxData OBJShape::getBBox() const {
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


Vector3f OBJShape::getOrigin() const {
    Vector3f origin = {0,0,0};
    for ( auto& triangle: triangles ) {
        origin = origin + triangle.getOrigin();
    }
    return origin / (float) triangles.size();
}

bool OBJShape::isContainPoint( const Vector3f& p ) const {
    return true;
}

IntersectionData OBJShape::intersectsWithRay( const Ray& ray ) const {
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

Vector3f OBJShape::getNormal( const Vector3f& p ) const {
    return {};
}