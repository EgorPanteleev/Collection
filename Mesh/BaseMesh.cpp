//
// Created by auser on 5/12/24.
//

#include "BaseMesh.h"

__host__ __device__ BaseMesh::BaseMesh(): material() {}

__host__ __device__ void BaseMesh::setMaterial( const Material& _material ) {
    material = _material;
}

__host__ __device__ Material BaseMesh::getMaterial() const {
    return material;
}


void BaseMesh::loadMesh( const std::string& path ) {
    //TODO
    //Not impemented yet
}
__host__ __device__ Vector <Triangle> BaseMesh::getTriangles() {
    return {};
}