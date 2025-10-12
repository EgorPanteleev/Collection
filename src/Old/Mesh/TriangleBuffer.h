//
// Created by auser on 11/3/24.
//

#ifndef COLLECTION_VERTEXBUFFER_H
#define COLLECTION_VERTEXBUFFER_H
#include "Vec3.h"
#include "Vec4.h"
#include <Vector.h>
#include <Material.h>
#include <BBox.h>
#include <Ray.h>
//class Vertex {
//public:
//    Vec3d position;
//    //Vec2d texCoords
//};


//Triangles();
//void addTriangle( const Vec3d& v1, const Vec3d& v2, const Vec3d& v3 );
//size_t size() const;
//BBox getBBox() const;
//BBox getBBox( unsigned int index ) const;
//Vec3d getOrigin( unsigned int index ) const;
//double intersectsWithRay( const Ray& ray, unsigned int index ) const;
//
//public:
//int addVertex( const Vec3d& v1 );

//FUTURE - TriangleBuffer -> Mesh
class TriangleBuffer {
public:
    TriangleBuffer();
    void addTriangle( const Material& mat, const Vec3d& v1, const Vec3d& v2, const Vec3d& v3 );
    Vec3d getOrigin( size_t index ) const;
    BBox getBBox() const;
    BBox getBBox( size_t index ) const;
    Vec3d getNormal( size_t index ) const;
    double intersectsWithRay( const Ray& ray, size_t index ) const;
    size_t size() const;
public:
    size_t addVertex( const Vec3d& v1 );
    size_t addMaterial( const Material& mat );
    Vector<Vec3d> vertices;
    Vector<Vec4i> indices;
    Vector<Material> materials;
};


#endif //COLLECTION_VERTEXBUFFER_H
