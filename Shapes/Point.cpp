#include "Point.h"
void Point::setX( double _x ) {
    x = _x;
}

void Point::setY( double _y ) {
    y = _y;
}

void Point::setZ( double _z ) {
    z = _z;
}

double Point::getX() const {
    return x;
}

double Point::getY() const {
    return y;
}

double Point::getZ() const {
    return z;
}

void Point::set( const Point& p ) {
    x = p.getX();
    y = p.getY();
    z = p.getZ();
}

void Point::operator=( const Point& p ) {
    set(p);
}

Point Point::operator+( const Point& p ) const {
    Point ret;
    ret.setX( x + p.getX() );
    ret.setY( y + p.getY() );
    ret.setZ( z + p.getZ() );
    return ret;
}

Point Point::operator-( const Point& p ) const {
    Point ret;
    ret.setX( x - p.getX() );
    ret.setY( y - p.getY() );
    ret.setZ( z - p.getZ() );
    return ret;
}

Point Point::operator*( double a ) const {
    Point ret;
    ret.setX( x * a );
    ret.setY( y * a );
    ret.setZ( z * a );
    return ret;
}

Point Point::operator/( double a ) const {
    Point ret;
    ret.setX( x / a );
    ret.setY( y / a );
    ret.setZ( z / a );
    return ret;
}

bool Point::operator==( const Point& p ) {
    return ( x == p.getX() && y == p.getY() && z == p.getZ() );
}

bool Point::operator!=( const Point& p ) {
    return (!(*this == p));
}

Point::~Point() {
    x = 0;
    y = 0;
    z = 0;
}
Point::Point( const Point& p ) {
    set(p);
}