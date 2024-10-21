//
// Created by auser on 9/2/24.
//

#ifndef COLLECTION_TRIANGLES_H
#define COLLECTION_TRIANGLES_H
#include "Vector.h"
#include "Vec3.h"
#include "BBox.h"
#include "Ray.h"
//TODO change Vec3d to Vector3i
class Triangles {
public:
    Triangles();
    void addTriangle( const Vec3d& v1, const Vec3d& v2, const Vec3d& v3 );
    size_t size() const;
    BBox getBBox() const;
    BBox getBBox( unsigned int index ) const;
    Vec3d getOrigin( unsigned int index ) const;
    double intersectsWithRay( const Ray& ray, unsigned int index ) const;

public:
    int addVertex( const Vec3d& v1 );
    Vector<Vec3d> vertices;
    Vector<Vec3i> indices;
};


#endif //COLLECTION_TRIANGLES_H
