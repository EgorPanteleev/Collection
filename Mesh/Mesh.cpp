//
// Created by auser on 5/12/24.
//

#include <cmath>
#include "Mesh.h"
#include "OBJLoader.h"
#include "Random.h"

Mesh::Mesh(): material(), primitives() {}

void Mesh::setMaterial(const Material& _material ) {
    material = _material;
    for ( auto primitive : primitives )
        primitive->setMaterial( material );
}

Material Mesh::getMaterial() const {
    return material;
}

Vec3d Mesh::getSamplePoint() {
    int size = (int) primitives.size();
    if ( size == 0 ) return {};
    int ind = std::floor( randomDouble() * ( size - 1 ) );
    return primitives[ind]->getSamplePoint();
}

void Mesh::loadMesh(const std::string& path ) {
    OBJLoader::load( path, this );
}

void Mesh::rotate(const Vec3d& axis, double angle, bool group ) {
    Vec3d origin = getOrigin();
    if ( !group ) move( origin * ( -1 ) );
    for ( auto primitive: primitives ) {
        primitive->rotate( axis, angle );
    }
    if ( !group ) move( origin );
}

void Mesh::move(const Vec3d& p ) {
    for ( auto primitive: primitives ) {
        primitive->move( p );
    }
}

void Mesh::moveTo(const Vec3d& point ) {
    move( point - getOrigin() );
}

void Mesh::scale(double scaleValue, bool group ) {
    Vec3d oldOrigin = getOrigin();
    for ( auto primitive: primitives ) {
        primitive->scale( scaleValue );
    }
    if ( !group ) moveTo( oldOrigin );
}
void Mesh::scale(const Vec3d& scaleVec, bool group ) {
    Vec3d oldOrigin = getOrigin();
    for ( auto primitive: primitives ) {
        primitive->scale( scaleVec );
    }
    if ( !group ) moveTo( oldOrigin );
}

void Mesh::scaleTo(double scaleValue, bool group ) {
    BBox bbox = getBBox();
    Vec3d len = bbox.pMax - bbox.pMin;
    double maxLen = std::max ( std::max( len[0], len[1] ), len[2]);
    double cff = scaleValue / maxLen;
    scale( cff, group );
}

void Mesh::scaleTo(const Vec3d& scaleVec, bool group ) {
    BBox bbox = getBBox();
    Vec3d len = bbox.pMax - bbox.pMin;
    Vec3d cff = { scaleVec[0] / len[0], scaleVec[1] / len[1], scaleVec[2] / len[2] };
    scale( cff, group );
}

void Mesh::setMinPoint(const Vec3d& vec, int ind ) {
    Vec3d moveVec = vec - getBBox().pMin;
    for ( int i = 0; i < 3; ++i ) {
        if ( ind == -1 || ind == i ) continue;
        moveVec[i] = 0;
    }
    move( moveVec );
}

void Mesh::setMaxPoint(const Vec3d& vec, int ind ) {
    Vec3d moveVec = vec - getBBox().pMax;
    for ( int i = 0; i < 3; ++i ) {
        if ( ind == -1 || ind == i ) continue;
        moveVec[i] = 0;
    }
    move( moveVec );
}

Vector<Primitive*> Mesh::getPrimitives() {
    return primitives;
}

BBox Mesh::getBBox() const {
    Vec3d vMin = {__FLT_MAX__,__FLT_MAX__,__FLT_MAX__};
    Vec3d vMax = {__FLT_MIN__,__FLT_MIN__,__FLT_MIN__};
    for ( auto primitive: primitives ) {
        BBox bbox = primitive->getBBox();
        vMin = min( bbox.pMin, vMin );
        vMax = max( bbox.pMax, vMax );
    }
    return { vMin, vMax };
}


Vec3d Mesh::getOrigin() const {
    Vec3d origin = {0,0,0};
    for ( auto primitive: primitives ) {
        origin = origin + primitive->getOrigin();
    }
    return origin / primitives.size();
}

bool Mesh::isContainPoint(const Vec3d& p ) const {
    for ( const auto primitive: primitives ) {
        if ( primitive->isContainPoint( p ) ) return true;
    }
    return false;
}

//IntersectionData Mesh::intersectsWithRay(const Ray& ray ) const {
//    double min = __FLT_MAX__;
//    Vec3d N = {};
//    for ( const auto primitive: primitives ) {
//        double t = primitive->intersectsWithRay( ray );
//        if ( t >= min ) continue;
//        min = t;
//    }
//    return { min ...};
//}

void Mesh::setPrimitives(Vector<Primitive*>& _primitives ) {
    primitives = _primitives;
    for ( auto primitive: primitives )
        primitive->setMaterial( material );
}
void Mesh::addPrimitive( Primitive* primitive ) {
    primitives.push_back( primitive );
    primitives[ primitives.size() - 1 ]->setMaterial( material );
}
