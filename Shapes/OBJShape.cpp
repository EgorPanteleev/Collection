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
    return { min, N };
}

Vector3f OBJShape::getNormal( const Vector3f& p ) const {
    return {};
}