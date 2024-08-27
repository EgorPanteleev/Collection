//
// Created by auser on 8/26/24.
//
#include <cmath>
#include "GroupOfMeshes.h"
#include "OBJLoader.h"
#include "Utils.h"

GroupOfMeshes::GroupOfMeshes(): meshes() {}

void GroupOfMeshes::loadMesh(const std::string& path ) {
    OBJLoader::load( path, this );
}

void GroupOfMeshes::rotate(const Vector3f& axis, float angle ) {
    Vector3f origin = getOrigin();
    move( origin * ( -1 ) );
    for ( auto mesh: meshes ) {
        mesh->rotate( axis, angle, true );
    }
    move( origin );
}

void GroupOfMeshes::move(const Vector3f& p ) {
    for ( auto mesh: meshes ) {
        mesh->move( p );
    }
}

void GroupOfMeshes::moveTo(const Vector3f& point ) {
    move( point - getOrigin() );
}

void GroupOfMeshes::scale(float scaleValue ) {
    Vector3f oldOrigin = getOrigin();
    for ( auto mesh: meshes ) {
        mesh->scale( scaleValue, true );
    }
    moveTo( oldOrigin );
}
void GroupOfMeshes::scale(const Vector3f& scaleVec ) {
    Vector3f oldOrigin = getOrigin();
    for ( auto mesh: meshes ) {
        mesh->scale( scaleVec, true );
    }
    moveTo( oldOrigin );
}

void GroupOfMeshes::scaleTo(float scaleValue ) {
    BBox bbox = getBBox();
    Vector3f len = bbox.pMax - bbox.pMin;
    float maxLen = std::max ( std::max( len.getX(), len.getY() ), len.getZ());
    float cff = scaleValue / maxLen;
    scale( cff );
}

void GroupOfMeshes::scaleTo(const Vector3f& scaleVec ) {
    BBox bbox = getBBox();
    Vector3f len = bbox.pMax - bbox.pMin;
    Vector3f cff = { scaleVec[0] / len[0], scaleVec[1] / len[1], scaleVec[2] / len[2] };
    scale( cff );
}

void GroupOfMeshes::setMinPoint(const Vector3f& vec, int ind ) {
    Vector3f moveVec = vec - getBBox().pMin;
    for ( int i = 0; i < 3; ++i ) {
        if ( ind == -1 || ind == i ) continue;
        moveVec[i] = 0;
    }
    move( moveVec );
}

void GroupOfMeshes::setMaxPoint(const Vector3f& vec, int ind ) {
    Vector3f moveVec = vec - getBBox().pMax;
    for ( int i = 0; i < 3; ++i ) {
        if ( ind == -1 || ind == i ) continue;
        moveVec[i] = 0;
    }
    move( moveVec );
}

Vector<Mesh*> GroupOfMeshes::getMeshes() const {
    return meshes;
}

BBox GroupOfMeshes::getBBox() const {
    Vector3f vMin = {__FLT_MAX__,__FLT_MAX__,__FLT_MAX__};
    Vector3f vMax = {__FLT_MIN__,__FLT_MIN__,__FLT_MIN__};
    for ( auto mesh: meshes ) {
        BBox bbox = mesh->getBBox();
        vMin = min( bbox.pMin, vMin );
        vMax = max( bbox.pMax, vMax );
    }
    return { vMin, vMax };
}


Vector3f GroupOfMeshes::getOrigin() const {
    Vector3f origin = {0,0,0};
    for ( auto mesh: meshes ) {
        origin = origin + mesh->getOrigin();
    }
    return origin / (float) meshes.size();
}

void GroupOfMeshes::setMaterial( Material material, int index ) {
    meshes[ index ]->setMaterial( material );
}

void GroupOfMeshes::setMaterial( Material material ) {
    for ( auto mesh: meshes ) {
        mesh->setMaterial( material );
    }
}

void GroupOfMeshes::setMeshes( Vector<Mesh*>& _meshes ) {
    meshes = _meshes;
}

void GroupOfMeshes::addMesh( Mesh* mesh ) {
    meshes.push_back( mesh );
}