//
// Created by igor on 28.01.2024.
//

#include "Vector2f.h"
#include <cmath>
#include <optional>
void Vector2f::setX( float _x ) {
    x = _x;
}

void Vector2f::setY( float _y ) {
    y = _y;
}

float Vector2f::getX() const {
    return x;
}

float Vector2f::getY() const {
    return y;
}

void Vector2f::set( const Vector2f& p ) {
    x = p.getX();
    y = p.getY();
}

Vector2f Vector2f::normalize() const {
    Vector2f res;
    res = *this;
    float len = sqrt( pow( x, 2 ) +  pow( y, 2 ));
    return res/len;
}

float& Vector2f::operator[]( int index ) {
    if ( index == 0) return x;
    if ( index == 1) return y;
    return (float &) std::nullopt;
}

const float& Vector2f::operator[]( int index ) const {
    if ( index == 0) return x;
    if ( index == 1) return y;
    return (float &) std::nullopt;
}

Vector2f& Vector2f::operator=( const Vector2f& p ) {
    set(p);
    return *this;
}

Vector2f Vector2f::operator+( const Vector2f& p ) const {
    Vector2f ret;
    ret.setX( x + p.getX() );
    ret.setY( y + p.getY() );
    return ret;
}

Vector2f Vector2f::operator-( const Vector2f& p ) const {
    Vector2f ret;
    ret.setX( x - p.getX() );
    ret.setY( y - p.getY() );
    return ret;
}

Vector2f Vector2f::operator*( float a ) const {
    Vector2f ret;
    ret.setX( x * a );
    ret.setY( y * a );
    return ret;
}

Vector2f Vector2f::operator/( float a ) const {
    Vector2f ret;
    ret.setX( x / a );
    ret.setY( y / a );
    return ret;
}

bool Vector2f::operator==( const Vector2f& p ) const {
    return ( x == p.getX() && y == p.getY() );
}

bool Vector2f::operator!=( const Vector2f& p ) const {
    return (!(*this == p));
}

Vector2f::Vector2f( const Vector2f& p ) {
    set(p);
}

Vector2f::Vector2f(): x( 0 ), y( 0 ) { }

Vector2f::Vector2f(float _x, float _y ): x( _x ), y( _y ){ }

Vector2f::~Vector2f() {
    x = 0;
    y = 0;
}