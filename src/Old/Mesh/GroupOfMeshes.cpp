//
// Created by auser on 8/26/24.
//
#include <cmath>
#include "GroupOfMeshes.h"
#include "OBJLoader.h"


GroupOfMeshes::GroupOfMeshes(): meshes() {}

void GroupOfMeshes::loadMesh(const std::string& path ) {
    OBJLoader::load( path, this );
}

void GroupOfMeshes::rotate(const Vec3d& axis, double angle ) {
    Vec3d origin = getOrigin();
    move( origin * ( -1 ) );
    for ( auto mesh: meshes ) {
        mesh->rotate( axis, angle, true );
    }
    move( origin );
}

void GroupOfMeshes::move(const Vec3d& p ) {
    for ( auto mesh: meshes ) {
        mesh->move( p );
    }
}

void GroupOfMeshes::moveTo(const Vec3d& point ) {
    move( point - getOrigin() );
}

void GroupOfMeshes::scale(double scaleValue ) {
    Vec3d oldOrigin = getOrigin();
    for ( auto mesh: meshes ) {
        mesh->scale( scaleValue, true );
    }
    moveTo( oldOrigin );
}
void GroupOfMeshes::scale(const Vec3d& scaleVec ) {
    Vec3d oldOrigin = getOrigin();
    for ( auto mesh: meshes ) {
        mesh->scale( scaleVec, true );
    }
    moveTo( oldOrigin );
}

void GroupOfMeshes::scaleTo(double scaleValue ) {
    BBox bbox = getBBox();
    Vec3d len = bbox.pMax - bbox.pMin;
    double maxLen = std::max ( std::max( len[0], len[1] ), len[2] );
    double cff = scaleValue / maxLen;
    scale( cff );
}

void GroupOfMeshes::scaleTo(const Vec3d& scaleVec ) {
    BBox bbox = getBBox();
    Vec3d len = bbox.pMax - bbox.pMin;
    Vec3d cff = { scaleVec[0] / len[0], scaleVec[1] / len[1], scaleVec[2] / len[2] };
    scale( cff );
}

void GroupOfMeshes::setMinPoint(const Vec3d& vec, int ind ) {
    Vec3d moveVec = vec - getBBox().pMin;
    for ( int i = 0; i < 3; ++i ) {
        if ( ind == -1 || ind == i ) continue;
        moveVec[i] = 0;
    }
    move( moveVec );
}

void GroupOfMeshes::setMaxPoint(const Vec3d& vec, int ind ) {
    Vec3d moveVec = vec - getBBox().pMax;
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
    Vec3d vMin = {__FLT_MAX__,__FLT_MAX__,__FLT_MAX__};
    Vec3d vMax = {__FLT_MIN__,__FLT_MIN__,__FLT_MIN__};
    for ( auto mesh: meshes ) {
        BBox bbox = mesh->getBBox();
        vMin = min( bbox.pMin, vMin );
        vMax = max( bbox.pMax, vMax );
    }
    return { vMin, vMax };
}


Vec3d GroupOfMeshes::getOrigin() const {
    Vec3d origin = {0,0,0};
    for ( auto mesh: meshes ) {
        origin = origin + mesh->getOrigin();
    }
    return origin / (double) meshes.size();
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