#include "Vector.h"
void Vector3f::setX( float _x ) {
    x = _x;
}

void Vector3f::setY( float _y ) {
    y = _y;
}

void Vector3f::setZ( float _z ) {
    z = _z;
}

float Vector3f::getX() const {
    return x;
}

float Vector3f::getY() const {
    return y;
}

float Vector3f::getZ() const {
    return z;
}

void Vector3f::set( const Vector3f& p ) {
    x = p.getX();
    y = p.getY();
    z = p.getZ();
}

void Vector3f::operator=( const Vector3f& p ) {
    set(p);
}

Vector3f Vector3f::operator+( const Vector3f& p ) const {
    Vector3f ret;
    ret.setX( x + p.getX() );
    ret.setY( y + p.getY() );
    ret.setZ( z + p.getZ() );
    return ret;
}

Vector3f Vector3f::operator-( const Vector3f& p ) const {
    Vector3f ret;
    ret.setX( x - p.getX() );
    ret.setY( y - p.getY() );
    ret.setZ( z - p.getZ() );
    return ret;
}

Vector3f Vector3f::operator*( float a ) const {
    Vector3f ret;
    ret.setX( x * a );
    ret.setY( y * a );
    ret.setZ( z * a );
    return ret;
}

Vector3f Vector3f::operator/( float a ) const {
    Vector3f ret;
    ret.setX( x / a );
    ret.setY( y / a );
    ret.setZ( z / a );
    return ret;
}

bool Vector3f::operator==( const Vector3f& p ) {
    return ( x == p.getX() && y == p.getY() && z == p.getZ() );
}

bool Vector3f::operator!=( const Vector3f& p ) {
    return (!(*this == p));
}

Vector3f::~Vector3f() {
    x = 0;
    y = 0;
    z = 0;
}
Vector3f::Vector3f( const Vector3f& p ) {
    set(p);
}