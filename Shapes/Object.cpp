#include "Object.h"

Object::Object(): shape( nullptr ), material() {
}

Object::Object( Shape* shape, const Material& material ): shape( shape ), material( material ) {
}

Object::~Object() {
}

void Object::rotate( const Vector3f& axis, float angle ) {
    shape->rotate( axis, angle );
}

void Object::move( const Vector3f& p ) {
    shape->move( p );
}

Vector3f Object::getOrigin() const {
    return shape->getOrigin();
}

bool Object::isContainPoint( const Vector3f& p ) const {
    return shape->isContainPoint( p );
}

IntersectionData Object::intersectsWithRay( const Ray& ray ) const {
    return shape->intersectsWithRay( ray );
}

Vector3f Object::getNormal( const Vector3f& p ) const {
    return shape->getNormal( p );
}

RGB Object::getColor() const {
    return material.getColor();
}

void Object::setColor( const RGB& c ) {
    material.setColor( c );
}

float Object::getDiffuse() const {
    return material.getDiffuse();
}

void Object::setDiffuse( float d ) {
    material.setDiffuse( d );
}

float Object::getReflection() const {
    return material.getReflection();
}

void Object::setReflection( float r ) {
    material.setReflection( r );
}