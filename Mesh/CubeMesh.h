//
// Created by auser on 5/12/24.
//

#ifndef COLLECTION_CUBEMESH_H
#define COLLECTION_CUBEMESH_H

#include "Vector.h"
#include "Mesh.h"
#include "Vector.h"
#include "Triangle.h"

class CubeMesh: public Mesh {
public:
    CubeMesh();
    CubeMesh( const Vec3d& _p1, const Vec3d& _p2);
    CubeMesh( const Vec3d& _p1, const Vec3d& _p2, const Material& _material );
private:
    void fillTriangles();
    Vec3d p1;
    Vec3d p2;
};
#endif //COLLECTION_CUBEMESH_H