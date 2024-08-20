////
//// Created by auser on 8/20/24.
////
//
//#include "Primitive.h"
//
//Primitive::Primitive(): triangle( nullptr ), sphere( nullptr ), type( UNKNOWN ){
//}
//Primitive::Primitive( Triangle* tri ): triangle( tri ), sphere( nullptr ), type( TRIANGLE ) {
//}
//Primitive::Primitive( Sphere* sph ): triangle( nullptr ), sphere( sph ), type( SPHERE ) {
//}
//Primitive::Type Primitive::getType() const {
//    return type;
//}
//
//bool Primitive::operator==( const Primitive& other ) {
//    switch (type) {
//        case TRIANGLE: return triangle == other.triangle;
//        case SPHERE: return sphere == other.sphere;
//        case UNKNOWN: return false;
//    }
//}
//
//bool Primitive::operator!=( const Primitive& other ) {
//    switch (type) {
//        case TRIANGLE: return triangle != other.triangle;
//        case SPHERE: return sphere != other.sphere;
//        case UNKNOWN: return true;
//    }
//}
//
//void Primitive::rotate( const Vector3f& axis, float angle ) {
//    switch (type) {
//        case TRIANGLE: return triangle->rotate( axis, angle );
//        case SPHERE: return sphere->rotate( axis, angle );
//        case UNKNOWN: return;
//    }
//}
//void Primitive::move( const Vector3f& p ) {
//    switch (type) {
//        case TRIANGLE: return triangle->move( p );
//        case SPHERE: return sphere->move( p );
//        case UNKNOWN: return;
//    }
//}
//void Primitive::moveTo( const Vector3f& point ) {
//    switch (type) {
//        case TRIANGLE: return triangle->moveTo( point );
//        case SPHERE: return sphere->moveTo( point );
//        case UNKNOWN: return;
//    }
//}
//void Primitive::scale( float scaleValue ) {
//    switch (type) {
//        case TRIANGLE: return triangle->scale( scaleValue );
//        case SPHERE: return sphere->scale( scaleValue );
//        case UNKNOWN: return;
//    }
//}
//void Primitive::scale( const Vector3f& scaleVec ) {
//    switch (type) {
//        case TRIANGLE: return triangle->scale( scaleVec );
//        case SPHERE: return sphere->scale( scaleVec );
//        case UNKNOWN: return;
//    }
//}
//void Primitive::scaleTo( float scaleValue ) {
//    switch (type) {
//        case TRIANGLE: return triangle->scaleTo( scaleValue );
//        case SPHERE: return sphere->scaleTo( scaleValue );
//        case UNKNOWN: return;
//    }
//}
//void Primitive::scaleTo( const Vector3f& scaleVec ) {
//    switch (type) {
//        case TRIANGLE: return triangle->scaleTo( scaleVec );
//        case SPHERE: return sphere->scaleTo( scaleVec );
//        case UNKNOWN: return;
//    }
//}
//void Primitive::setMaterial( const Material& mat ) {
//    switch (type) {
//        case TRIANGLE: return triangle->setMaterial( mat );
//        case SPHERE: return sphere->setMaterial( mat );
//        case UNKNOWN: return;
//    }
//}
//
//Vector3f Primitive::getSamplePoint() const {
//    switch (type) {
//        case TRIANGLE: return triangle->getSamplePoint();
//        case SPHERE: return sphere->getSamplePoint();
//        case UNKNOWN: return {};
//    }
//}
//BBox Primitive::getBBox() const {
//    switch (type) {
//        case TRIANGLE: return triangle->getBBox();
//        case SPHERE: return sphere->getBBox();
//        case UNKNOWN: return {};
//    }
//}
//Vector3f Primitive::getOrigin() const {
//    switch (type) {
//        case TRIANGLE: return triangle->getOrigin();
//        case SPHERE: return sphere->getOrigin();
//        case UNKNOWN: return {};
//    }
//}
//bool Primitive::isContainPoint( const Vector3f& p ) const {
//    switch (type) {
//        case TRIANGLE: return triangle->isContainPoint( p );
//        case SPHERE: return sphere->isContainPoint( p );
//        case UNKNOWN: return {};
//    }
//}
//float Primitive::intersectsWithRay( const Ray& ray ) const {
//    switch (type) {
//        case TRIANGLE: return triangle->intersectsWithRay( ray );
//        case SPHERE: return sphere->intersectsWithRay( ray );
//        case UNKNOWN: return {};
//    }
//}
//Vector3f Primitive::getNormal( const Vector3f& P ) const {
//    switch (type) {
//        case TRIANGLE: return triangle->getNormal( P );
//        case SPHERE: return sphere->getNormal( P );
//        case UNKNOWN: return {};
//    }
//}
//RGB Primitive::getColor( const Vector3f& P ) const {
//    switch (type) {
//        case TRIANGLE: return triangle->getColor( P );
//        case SPHERE: return sphere->getColor( P );
//        case UNKNOWN: return {};
//    }
//}
//RGB Primitive::getAmbient( const Vector3f& P ) const {
//    switch (type) {
//        case TRIANGLE: return triangle->getAmbient( P );
//        case SPHERE: return sphere->getAmbient( P );
//        case UNKNOWN: return {};
//    }
//}
//float Primitive::getRoughness( const Vector3f& P ) const {
//    switch (type) {
//        case TRIANGLE: return triangle->getRoughness( P );
//        case SPHERE: return sphere->getRoughness( P );
//        case UNKNOWN: return {};
//    }
//}
//
//Material Primitive::getMaterial() const {
//    switch (type) {
//        case TRIANGLE: return triangle->getMaterial();
//        case SPHERE: return sphere->getMaterial();
//        case UNKNOWN: return {};
//    }
//}