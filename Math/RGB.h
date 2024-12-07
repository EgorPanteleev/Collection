//
// Created by auser on 10/20/24.
//

#ifndef MATH_RGB_H
#define MATH_RGB_H
#undef RGB
#define BLACK RGB( 20, 20, 20 )
#define WHITE RGB( 255, 255, 255 )
#define GRAY RGB( 210, 210, 210 )
#define RED RGB( 255, 0, 0 )
#define GREEN RGB( 0, 255, 0 )
#define BLUE RGB( 0, 0, 255 )
#define YELLOW RGB( 255, 255, 0 )
#define BROWN RGB( 150, 75, 0 )
#define PINK RGB( 255,105,180 )
#define DARK_BLUE RGB(65,105,225)
#define CYAN RGB( 0, 255, 255)
#include <iostream>
#include "SystemUtils.h"

class RGB {
public:
    HOST_DEVICE RGB(): r(), g(), b() {};
    HOST_DEVICE RGB( double _r, double _g, double _b): r( _r ), g( _g ), b ( _b ) {
        if ( r > 255 ) r = 255;
        if ( g > 255 ) g = 255;
        if ( b > 255 ) b = 255;
    }
    HOST_DEVICE RGB( const RGB& other ) {
        r = other.r;
        g = other.g;
        b = other.b;
    }

    HOST_DEVICE RGB& operator=( const RGB& other ) {
        r = other.r;
        g = other.g;
        b = other.b;
        return *this;
    }

    HOST_DEVICE ~RGB() {}

    HOST_DEVICE RGB& operator+=( const RGB& other ) {
        r += other.r;
        g += other.g;
        b += other.b;
        return *this;
    }

    HOST_DEVICE RGB operator+( const RGB& other ) const {
        return { r + other.r, g + other.g, b + other.b };
    }

    HOST_DEVICE RGB& operator*=( const RGB& other ) {
        r *= other.r;
        g *= other.g;
        b *= other.b;
        return *this;
    }

    HOST_DEVICE RGB operator*( const RGB& other ) const {
        return { r * other.r, g * other.g, b * other.b };
    }

    HOST_DEVICE RGB& operator/=( const RGB& other ) {
        r /= other.r;
        g /= other.g;
        b /= other.b;
        return *this;
    }

    HOST_DEVICE RGB operator/( const RGB& other ) const {
        return { r / other.r, g / other.g, b / other.b };
    }

    HOST_DEVICE bool operator==( const RGB& other ) const {
        return r == other.r && g == other.g && b == other.b;
    }
    HOST_DEVICE bool operator!=( const RGB& other ) const {
        return !( *this == other );
    }

    HOST_DEVICE void scaleTo( double value ) {
        double max = std::max( std::max( r, g ), b );
        if ( max < value ) return;
        r = ( r / max ) * value;
        g = ( g / max ) * value;
        b = ( b / max ) * value;
    }


public:
    double r;
    double g;
    double b;
};

HOST_DEVICE inline RGB operator+( const RGB& col, const double& d ) {
    return { col.r + d, col.g + d, col.b + d };
}

HOST_DEVICE inline RGB operator+( const double& d, const RGB& col ) {
    return { col.r + d, col.g + d, col.b + d };
}

HOST_DEVICE inline RGB operator*( const RGB& col, const double& d ) {
    return { col.r * d, col.g * d, col.b * d };
}

HOST_DEVICE inline RGB operator*( const double& d, const RGB& col ) {
    return { col.r * d, col.g * d, col.b * d };
}

HOST_DEVICE inline RGB operator/( const RGB& col, const double& d ) {
    return { col.r / d, col.g / d, col.b / d };
}

HOST_DEVICE inline RGB operator/( const double& d, const RGB& col ) {
    return { col.r / d, col.g / d, col.b / d };
}



#endif //MATH_RGB_H
