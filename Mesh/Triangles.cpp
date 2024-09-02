//
// Created by auser on 9/2/24.
//

#include "Triangles.h"
#include "Utils.h"
Triangles::Triangles(): vertices(), indices() {
}

int Triangles::addVertex( const Vector3f& v1 ) {
//    int index = vertices.find( v1 );
//    if ( index == -1 ) {
//        index = vertices.size();
//        vertices.push_back( v1 );
//    }
    vertices.push_back( v1 );
    return vertices.size() - 1/*index*/;
}

void Triangles::addTriangle( const Vector3f& v1, const Vector3f& v2, const Vector3f& v3 ) {
    indices.push_back( { (float) addVertex( v1 ), (float) addVertex( v2 ), (float) addVertex( v3 ) } );
}

size_t Triangles::size() const {
    return indices.size();
}

BBox Triangles::getBBox() const {
    Vector3f vMin = {__FLT_MAX__,__FLT_MAX__,__FLT_MAX__};
    Vector3f vMax = {-__FLT_MAX__,-__FLT_MAX__,-__FLT_MAX__};
    for ( const auto& ind: indices ) {
        vMin = min( vMin, vertices[ (int) ind[0] ] );
        vMin = min( vMin, vertices[ (int) ind[1] ] );
        vMin = min( vMin, vertices[ (int) ind[2] ] );
        vMax = max( vMax, vertices[ (int) ind[0] ] );
        vMax = max( vMax, vertices[ (int) ind[1] ] );
        vMax = max( vMax, vertices[ (int) ind[2] ] );
    }
    return { vMin, vMax };
}

BBox Triangles::getBBox( unsigned int index ) const {
    Vector3f vMin = {__FLT_MAX__,__FLT_MAX__,__FLT_MAX__};
    Vector3f vMax = {-__FLT_MAX__,-__FLT_MAX__,-__FLT_MAX__};
    Vector3f ind = indices[ index ];

    vMin = min( vMin, vertices[ (int) ind[0] ] );
    vMin = min( vMin, vertices[ (int) ind[1] ] );
    vMin = min( vMin, vertices[ (int) ind[2] ] );
    vMax = max( vMax, vertices[ (int) ind[0] ] );
    vMax = max( vMax, vertices[ (int) ind[1] ] );
    vMax = max( vMax, vertices[ (int) ind[2] ] );

    return { vMin, vMax };
}

Vector3f Triangles::getOrigin( unsigned int index ) const {
    Vector3f ind = indices[ index ];
    return (  vertices[ (int) ind[0] ] +  vertices[ (int) ind[1] ] +  vertices[ (int) ind[2] ] ) / 3;
}

float Triangles::intersectsWithRay( const Ray& ray, unsigned int index ) const {
    Vector3f ind = indices[ index ];
    Vector3f edge1 = vertices[ (int) ind[1] ] - vertices[ (int) ind[0] ];
    Vector3f edge2 = vertices[ (int) ind[2] ] - vertices[ (int) ind[0] ];
    Vector3f h = ray.direction.cross( edge2 );
    float a = dot(edge1, h);

    if ( a < __FLT_EPSILON__ ) return __FLT_MAX__;

    float f = 1.0f / a;
    Vector3f s = ray.origin - vertices[ (int) ind[0] ];
    float u = f * dot(s, h);

    if ( u < 0.0f || u > 1.0f ) return __FLT_MAX__;

    Vector3f q = s.cross( edge1 );
    float v = f * dot(ray.direction, q);

    if  ( v < 0.0f || u + v > 1.0f ) return __FLT_MAX__;

    float t = f * dot(edge2, q);

    if ( t < __FLT_EPSILON__ ) return __FLT_MAX__;

    return t;
}

