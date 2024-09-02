//
// Created by auser on 9/2/24.
//

#ifndef COLLECTION_TRIANGLES_H
#define COLLECTION_TRIANGLES_H
#include "Vector.h"
#include "Vector3f.h"
#include "BBox.h"
#include "Ray.h"
//TODO change Vector3f to Vector3i
class Triangles {
public:
    Triangles();
    void addTriangle( const Vector3f& v1, const Vector3f& v2, const Vector3f& v3 );
    size_t size() const;
    BBox getBBox() const;
    BBox getBBox( unsigned int index ) const;
    Vector3f getOrigin( unsigned int index ) const;
    float intersectsWithRay( const Ray& ray, unsigned int index ) const;

public:
    int addVertex( const Vector3f& v1 );
    Vector<Vector3f> vertices;
    Vector<Vector3f> indices;
};


#endif //COLLECTION_TRIANGLES_H
