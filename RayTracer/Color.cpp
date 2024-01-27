#include "Color.h"


void RGB::set( float _r, float _g, float _b ) {
    r = _r;
    g = _g;
    b = _b;
}
RGB RGB::operator*( double a) const {
    return { r * a, g * a, b * a };
}
RGB RGB::operator/( double a) const {
    return { r / a, g / a, b / a };
}
RGB::RGB(): r(0), g(0), b(0) {
}
RGB::RGB( double _r, double _g, double _b): r(_r), g(_g), b(_b) {
}
RGB::~RGB() {
    r = 0;
    g = 0;
    b = 0;
}
