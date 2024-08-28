//
// Created by auser on 5/12/24.
//

#include <cmath>
#include "Mesh.h"
#include "OBJLoader.h"
#include "Utils.h"

Mesh::Mesh(): material(), primitives() {}

void Mesh::setMaterial(const Material& _material ) {
    material = _material;
    for ( auto primitive : primitives )
        primitive->setMaterial( material );
}

Material Mesh::getMaterial() const {
    return material;
}

Vector3f Mesh::getSamplePoint() {
    int size = (int) primitives.size();
    if ( size == 0 ) return {};
    int ind = std::floor( randomFloat() * ( (float) size - 1 ) );
    return primitives[ind]->getSamplePoint();
}

void Mesh::loadMesh(const std::string& path ) {
    OBJLoader::load( path, this );
}

void Mesh::rotate(const Vector3f& axis, float angle, bool group ) {
    Vector3f origin = getOrigin();
    if ( !group ) move( origin * ( -1 ) );
    for ( auto primitive: primitives ) {
        primitive->rotate( axis, angle );
    }
    if ( !group ) move( origin );
}

void Mesh::move(const Vector3f& p ) {
    for ( auto primitive: primitives ) {
        primitive->move( p );
    }
}

void Mesh::moveTo(const Vector3f& point ) {
    move( point - getOrigin() );
}

void Mesh::scale(float scaleValue, bool group ) {
    Vector3f oldOrigin = getOrigin();
    for ( auto primitive: primitives ) {
        primitive->scale( scaleValue );
    }
    if ( !group ) moveTo( oldOrigin );
}
void Mesh::scale(const Vector3f& scaleVec, bool group ) {
    Vector3f oldOrigin = getOrigin();
    for ( auto primitive: primitives ) {
        primitive->scale( scaleVec );
    }
    if ( !group ) moveTo( oldOrigin );
}

void Mesh::scaleTo(float scaleValue, bool group ) {
    BBox bbox = getBBox();
    Vector3f len = bbox.pMax - bbox.pMin;
    float maxLen = std::max ( std::max( len.getX(), len.getY() ), len.getZ());
    float cff = scaleValue / maxLen;
    scale( cff, group );
}

void Mesh::scaleTo(const Vector3f& scaleVec, bool group ) {
    BBox bbox = getBBox();
    Vector3f len = bbox.pMax - bbox.pMin;
    Vector3f cff = { scaleVec[0] / len[0], scaleVec[1] / len[1], scaleVec[2] / len[2] };
    scale( cff, group );
}

void Mesh::setMinPoint(const Vector3f& vec, int ind ) {
    Vector3f moveVec = vec - getBBox().pMin;
    for ( int i = 0; i < 3; ++i ) {
        if ( ind == -1 || ind == i ) continue;
        moveVec[i] = 0;
    }
    move( moveVec );
}

void Mesh::setMaxPoint(const Vector3f& vec, int ind ) {
    Vector3f moveVec = vec - getBBox().pMax;
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
    Vector3f vMin = {__FLT_MAX__,__FLT_MAX__,__FLT_MAX__};
    Vector3f vMax = {__FLT_MIN__,__FLT_MIN__,__FLT_MIN__};
    for ( auto primitive: primitives ) {
        BBox bbox = primitive->getBBox();
        vMin = min( bbox.pMin, vMin );
        vMax = max( bbox.pMax, vMax );
    }
    return { vMin, vMax };
}


Vector3f Mesh::getOrigin() const {
    Vector3f origin = {0,0,0};
    for ( auto primitive: primitives ) {
        origin = origin + primitive->getOrigin();
    }
    return origin / (float) primitives.size();
}

bool Mesh::isContainPoint(const Vector3f& p ) const {
    for ( const auto primitive: primitives ) {
        if ( primitive->isContainPoint( p ) ) return true;
    }
    return false;
}

IntersectionData Mesh::intersectsWithRay(const Ray& ray ) const {
    float min = __FLT_MAX__;
    Vector3f N = {};
    for ( const auto primitive: primitives ) {
        float t = primitive->intersectsWithRay( ray );
        if ( t >= min ) continue;
        min = t;
    }
    return { min, nullptr };
}

void Mesh::setPrimitives(Vector<Primitive*>& _primitives ) {
    primitives = _primitives;
    for ( auto primitive: primitives )
        primitive->setMaterial( material );
}
void Mesh::addPrimitive( Primitive* primitive ) {
    primitives.push_back( primitive );
    primitives[ primitives.size() - 1 ]->setMaterial( material );
}
