//
// Created by auser on 5/12/24.
//

#include "BaseMesh.h"

BaseMesh::BaseMesh(): material() {}

void BaseMesh::setMaterial( const Material& _material ) {
    material = _material;
}

Material BaseMesh::getMaterial() const {
    return material;
}


void BaseMesh::loadMesh( const std::string& path ) {
    //TODO
    //Not impemented yet
}
Vector <Triangle> BaseMesh::getTriangles() {
    return {};
}