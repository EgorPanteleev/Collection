#pragma once

class Vector4f {
public:
    void setX( float _x );

    void setY( float _y );

    void setZ( float _z );

    void setW( float _w );

    float getX() const;

    float getY() const;

    float getZ() const;

    float getW() const;

    void set( const Vector4f& p );
    //operators

    void operator=( const Vector4f& p );

    Vector4f normalize();

    Vector4f cross( Vector4f vec );

    float operator[]( int index );

    const float operator[]( int index ) const;

    Vector4f operator+( const Vector4f& p ) const;

    Vector4f operator-( const Vector4f& p ) const;

    Vector4f operator*( float a ) const;

    Vector4f operator/( float a ) const;

    bool operator==( const Vector4f& p );

    bool operator!=( const Vector4f& p );
    Vector4f();
    Vector4f(float _x, float _y, float _z, float _w);
    ~Vector4f();
    Vector4f( const Vector4f& p );
private:
    float x;
    float y;
    float z;
    float w;
};



