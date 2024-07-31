
#include "Color.h"
#include "cmath"

__host__ __device__ void RGB::set( float _r, float _g, float _b ) {
    //*this = RGB( _r, _g, _b );
}

__host__ __device__ RGB RGB::operator+( const RGB& color ) const {
    return { r + color.r, g + color.g, b + color.b };
}

__host__ __device__ RGB RGB::operator*( float a) const {
    return { r * a, g * a, b * a };
}
__host__ __device__ RGB RGB::operator/( float a) const {
    return { r / a, g / a, b / a };
}
__host__ __device__ bool RGB::operator==( const RGB& color ) const {
    return r == color.r && g == color.g && b == color.b;
}

__host__ __device__ RGB::RGB(): r(0), g(0), b(0) {
}
__host__ __device__ RGB::RGB( float _r, float _g, float _b): r(_r), g(_g), b(_b) {
    scaleTo( 255 );
}

__host__ __device__ void RGB::scaleTo( float value ) {
    float max = std::max ( std::max( r, g), b );
    if ( max < value ) return;
    r = ( r / max ) * value;
    g = ( g / max ) * value;
    b = ( b / max ) * value;
}

__host__ __device__ RGB::~RGB() {
    r = 0;
    g = 0;
    b = 0;
}
