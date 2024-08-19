
#include "Color.h"
#include "cmath"

void RGB::set( float _r, float _g, float _b ) {
    //*this = RGB( _r, _g, _b );
}

RGB RGB::operator+( const RGB& color ) const {
    return { r + color.r, g + color.g, b + color.b };
}

RGB RGB::operator*( float a) const {
    return { r * a, g * a, b * a };
}
RGB RGB::operator/( float a) const {
    return { r / a, g / a, b / a };
}
bool RGB::operator==( const RGB& color ) const {
    return r == color.r && g == color.g && b == color.b;
}

RGB::RGB(): r(0), g(0), b(0) {
}
RGB::RGB( float _r, float _g, float _b): r(_r), g(_g), b(_b) {
    if ( r > 255 ) r = 255;
    if ( g > 255 ) g = 255;
    if ( b > 255 ) b = 255;
}

void RGB::scaleTo( float value ) {
    float max = std::max ( std::max( r, g), b );
    if ( max < value ) return;
    r = ( r / max ) * value;
    g = ( g / max ) * value;
    b = ( b / max ) * value;
}

[[nodiscard]] Vector3f RGB::toNormal() const {
    Vector3f res = { r, g, b };
    res = res / 255 * 2 - Vector3f( 1, 1, 1 );
    return res.normalize();
}

RGB::~RGB() {
    r = 0;
    g = 0;
    b = 0;
}
