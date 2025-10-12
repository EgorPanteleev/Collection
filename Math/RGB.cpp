////
//// Created by auser on 10/20/24.
////
//
//#include "RGB.h"
//
////HOST_DEVICE RGB::RGB(): r(), g(), b() {};
//
//
////HOST_DEVICE RGB::RGB( double _r, double _g, double _b): r( _r ), g( _g ), b ( _b ) {
////    if ( r > 255 ) r = 255;
////    if ( g > 255 ) g = 255;
////    if ( b > 255 ) b = 255;
////}
////HOST_DEVICE RGB::RGB( const RGB& other ) {
////    r = other.r;
////    g = other.g;
////    b = other.b;
////}
//
////HOST_DEVICE RGB& RGB::operator=( const RGB& other ) {
////    r = other.r;
////    g = other.g;
////    b = other.b;
////    return *this;
////}
////
////HOST_DEVICE RGB::~RGB() {}
////
////HOST_DEVICE RGB& RGB::operator+=( const RGB& other ) {
////    r += other.r;
////    g += other.g;
////    b += other.b;
////    return *this;
////}
////
////HOST_DEVICE RGB RGB::operator+( const RGB& other ) const {
////    return { r + other.r, g + other.g, b + other.b };
////}
////
////HOST_DEVICE RGB& RGB::operator*=( const RGB& other ) {
////    r *= other.r;
////    g *= other.g;
////    b *= other.b;
////    return *this;
////}
////
////HOST_DEVICE RGB RGB::operator*( const RGB& other ) const {
////    return { r * other.r, g * other.g, b * other.b };
////}
////
////HOST_DEVICE RGB& RGB::operator/=( const RGB& other ) {
////    r /= other.r;
////    g /= other.g;
////    b /= other.b;
////    return *this;
////}
////
////HOST_DEVICE RGB RGB::operator/( const RGB& other ) const {
////    return { r / other.r, g / other.g, b / other.b };
////}
////
////HOST_DEVICE bool RGB::operator==( const RGB& other ) const {
////    return r == other.r && g == other.g && b == other.b;
////}
////HOST_DEVICE bool RGB::operator!=( const RGB& other ) const {
////    return !( *this == other );
////}
////
////HOST_DEVICE void RGB::scaleTo( double value ) {
////    double max = std::max( std::max( r, g ), b );
////    if ( max < value ) return;
////    r = ( r / max ) * value;
////    g = ( g / max ) * value;
////    b = ( b / max ) * value;
////}
//
//
