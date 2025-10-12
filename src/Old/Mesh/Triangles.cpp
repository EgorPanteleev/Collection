//
// Created by auser on 9/2/24.
//

#include "Triangles.h"

Triangles::Triangles(): vertices(), indices() {
}

int Triangles::addVertex( const Vec3d& v1 ) {
//    int index = vertices.find( v1 );
//    if ( index == -1 ) {
//        index = vertices.size();
//        vertices.push_back( v1 );
//    }
    vertices.push_back( v1 );
    return vertices.size() - 1/*index*/;
}

void Triangles::addTriangle( const Vec3d& v1, const Vec3d& v2, const Vec3d& v3 ) {
    indices.push_back( { addVertex( v1 ), addVertex( v2 ), addVertex( v3 ) } );
}

size_t Triangles::size() const {
    return indices.size();
}

BBox Triangles::getBBox() const {
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

BBox Triangles::getBBox( unsigned int index ) const {
    Vec3d vMin = {__FLT_MAX__,__FLT_MAX__,__FLT_MAX__};
    Vec3d vMax = {-__FLT_MAX__,-__FLT_MAX__,-__FLT_MAX__};
    Vec3i ind = indices[ index ];

    vMin = min( vMin, vertices[ (int) ind[0] ] );
    vMin = min( vMin, vertices[ (int) ind[1] ] );
    vMin = min( vMin, vertices[ (int) ind[2] ] );
    vMax = max( vMax, vertices[ (int) ind[0] ] );
    vMax = max( vMax, vertices[ (int) ind[1] ] );
    vMax = max( vMax, vertices[ (int) ind[2] ] );

    return { vMin, vMax };
}

Vec3d Triangles::getOrigin( unsigned int index ) const {
    Vec3i ind = indices[ index ];
    return (  vertices[ (int) ind[0] ] +  vertices[ (int) ind[1] ] +  vertices[ (int) ind[2] ] ) / 3;
}

double Triangles::intersectsWithRay( const Ray& ray, unsigned int index ) const {
    Vec3i ind = indices[ index ];
    Vec3d edge1 = vertices[ (int) ind[1] ] - vertices[ (int) ind[0] ];
    Vec3d edge2 = vertices[ (int) ind[2] ] - vertices[ (int) ind[0] ];
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

