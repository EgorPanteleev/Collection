//
// Created by auser on 11/3/24.
//

#include "TriangleBuffer.h"


TriangleBuffer::TriangleBuffer(): vertices(), indices(), materials() {
}

size_t TriangleBuffer::addVertex( const Vec3d& v1 ) {
//    int index = vertices.find( v1 );
//    if ( index == -1 ) {
//        index = vertices.size();
//        vertices.push_back( v1 );
//    }
    vertices.push_back( v1 );
    return vertices.size() - 1/*index*/;
}

size_t TriangleBuffer::addMaterial( const Material& mat ) {
    materials.push_back( mat );
    return materials.size() - 1;
}

void TriangleBuffer::addTriangle( const Material& mat, const Vec3d& v1, const Vec3d& v2, const Vec3d& v3 ) {
    indices.emplace_back( addVertex( v1 ), addVertex( v2 ), addVertex( v3 ), addMaterial( mat ) );
}

size_t TriangleBuffer::size() const {
    return indices.size();
}

BBox TriangleBuffer::getBBox() const {
    Vec3d vMin = {__FLT_MAX__,__FLT_MAX__,__FLT_MAX__};
    Vec3d vMax = {-__FLT_MAX__,-__FLT_MAX__,-__FLT_MAX__};
    for ( const auto& ind: indices ) {
        vMin = min( vMin, vertices[ ind[0] ] );
        vMin = min( vMin, vertices[ ind[1] ] );
        vMin = min( vMin, vertices[ ind[2] ] );
        vMax = max( vMax, vertices[ ind[0] ] );
        vMax = max( vMax, vertices[ ind[1] ] );
        vMax = max( vMax, vertices[ ind[2] ] );
    }
    return { vMin, vMax };
}

BBox TriangleBuffer::getBBox( size_t index ) const {
    Vec3d vMin = {__FLT_MAX__,__FLT_MAX__,__FLT_MAX__};
    Vec3d vMax = {-__FLT_MAX__,-__FLT_MAX__,-__FLT_MAX__};
    Vec4i ind = indices[ index ];

    vMin = min( vMin, vertices[ ind[0] ] );
    vMin = min( vMin, vertices[ ind[1] ] );
    vMin = min( vMin, vertices[ ind[2] ] );
    vMax = max( vMax, vertices[ ind[0] ] );
    vMax = max( vMax, vertices[ ind[1] ] );
    vMax = max( vMax, vertices[ ind[2] ] );

    return { vMin, vMax };
}

Vec3d TriangleBuffer::getNormal( size_t index ) const {
    Vec4i ind = indices[ index ];

    Vec3d edge1 = vertices[ ind[1] ] - vertices[ ind[0] ];
    Vec3d edge2 = vertices[ ind[2] ] - vertices[ ind[0] ];
    return cross( edge1, edge2 ).normalize();
}

Vec3d TriangleBuffer::getOrigin( size_t index ) const {
    static constexpr double a3 = 1 / 3;
    Vec4i ind = indices[ index ];
    return (  vertices[ ind[0] ] +  vertices[ ind[1] ] +  vertices[ ind[2] ] ) * a3;
}

double TriangleBuffer::intersectsWithRay( const Ray& ray, size_t index ) const {
    Vec4i ind = indices[ index ];
    Vec3d edge1 = vertices[ ind[1] ] - vertices[ ind[0] ];
    Vec3d edge2 = vertices[ ind[2] ] - vertices[ ind[0] ];
    Vec3d h = cross( ray.direction, edge2 );
    double a = dot(edge1, h);

    if ( a < __FLT_EPSILON__ ) return __FLT_MAX__;

    double f = 1.0 / a;
    Vec3d s = ray.origin - vertices[ (int) ind[0] ];
    double u = f * dot(s, h);

    if ( u < 0.0 || u > 1.0 ) return __FLT_MAX__;

    Vec3d q = cross( s, edge1 );
    double v = f * dot(ray.direction, q);

    if  ( v < 0.0 || u + v > 1.0 ) return __FLT_MAX__;

    double t = f * dot(edge2, q);

    if ( t < __FLT_EPSILON__ ) return __FLT_MAX__;

    return t;
}

