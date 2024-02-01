#include "OBJShape.h"
#include "OBJLoader.h"
OBJShape::OBJShape( const std::string& path ) {
    OBJLoader::load( path, this );
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