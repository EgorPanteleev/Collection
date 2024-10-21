//
// Created by auser on 10/20/24.
//

#include "RGB.h"

RGB::RGB(): r(), g(), b() {};
RGB::RGB( double _r, double _g, double _b): r( _r ), g( _g ), b ( _b ) {
    if ( r > 255 ) r = 255;
    if ( g > 255 ) g = 255;
    if ( b > 255 ) b = 255;
}
RGB::RGB( const RGB& other ) {
    r = other.r;
    g = other.g;
    b = other.b;
}

RGB& RGB::operator=( const RGB& other ) {
    r = other.r;
    g = other.g;
    b = other.b;
    return *this;
}

RGB::~RGB() {}

RGB& RGB::operator+=( const RGB& other ) {
    r += other.r;
    g += other.g;
    b += other.b;
    return *this;
}

RGB RGB::operator+( const RGB& other ) const {
    return { r + other.r, g + other.g, b + other.b };
}

RGB& RGB::operator*=( const RGB& other ) {
    r *= other.r;
    g *= other.g;
    b *= other.b;
    return *this;
}

RGB RGB::operator*( const RGB& other ) const {
    return { r * other.r, g * other.g, b * other.b };
}

RGB& RGB::operator/=( const RGB& other ) {
    r /= other.r;
    g /= other.g;
    b /= other.b;
    return *this;
}

RGB RGB::operator/( const RGB& other ) const {
    return { r / other.r, g / other.g, b / other.b };
}

bool RGB::operator==( const RGB& other ) const {
    return r == other.r && g == other.g && b == other.b;
}
bool RGB::operator!=( const RGB& other ) const {
    return !( *this == other );
}

void RGB::scaleTo( double value ) {
    double max = std::max( std::max( r, g ), b );
    if ( max < value ) return;
    r = ( r / max ) * value;
    g = ( g / max ) * value;
    b = ( b / max ) * value;
}

RGB operator+( const RGB& col, const double& d ) {
    return { col.r + d, col.g + d, col.b + d };
}

RGB operator+( const double& d, const RGB& col ) {
    return { col.r + d, col.g + d, col.b + d };
}

RGB operator*( const RGB& col, const double& d ) {
    return { col.r * d, col.g * d, col.b * d };
}

RGB operator*( const double& d, const RGB& col ) {
    return { col.r * d, col.g * d, col.b * d };
}

RGB operator/( const RGB& col, const double& d ) {
    return { col.r / d, col.g / d, col.b / d };
}

RGB operator/( const double& d, const RGB& col ) {
    return { col.r / d, col.g / d, col.b / d };
}

