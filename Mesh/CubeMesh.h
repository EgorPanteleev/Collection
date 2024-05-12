//
// Created by auser on 5/12/24.
//

#ifndef COLLECTION_CUBEMESH_H
#define COLLECTION_CUBEMESH_H

#include <vector>
#include "TriangularMesh.h"
#include "Vector.h"

class CubeMesh: public TriangularMesh {
public:
    CubeMesh();
    CubeMesh( const Vector3f& _p1, const Vector3f& _p2);
    CubeMesh( const Vector3f& _p1, const Vector3f& _p2, const Material& _material );
private:
    void fillTriangles();
    Vector3f p1;
    Vector3f p2;
};
#endif //COLLECTION_CUBEMESH_H