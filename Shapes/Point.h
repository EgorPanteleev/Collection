#ifndef COLLECTION_POINT_H
#define COLLECTION_POINT_H


class Point {
public:
    void setX( double _x );

    void setY( double _y );

    void setZ( double _z );

    double getX() const;

    double getY() const;

    double getZ() const;

    void set( const Point& p );
    //operators

    void operator=( const Point& p );

    Point operator+( const Point& p ) const;

    Point operator-( const Point& p ) const;

    bool operator==( const Point& p );

    bool operator!=( const Point& p );
    Point(): x(0), y(0), z(0){ }
    Point(double _x, double _y, double _z): x(_x), y(_y), z(_z){ }
    ~Point();
    Point( const Point& p );
private:
    double x;
    double y;
    double z;
};


#endif //COLLECTION_POINT_H
