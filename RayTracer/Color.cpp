#include "Color.h"


void RGB::set( float _r, float _g, float _b ) {
    r = _r;
    g = _g;
    b = _b;
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
}
RGB::~RGB() {
    r = 0;
    g = 0;
    b = 0;
}
