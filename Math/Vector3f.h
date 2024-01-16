#pragma once

class Vector3f {
public:
    void setX( float _x );

    void setY( float _y );

    void setZ( float _z );

    float getX() const;

    float getY() const;

    float getZ() const;

    void set( const Vector3f& p );
    //operators

    void operator=( const Vector3f& p );

    Vector3f normalize();

    Vector3f cross( Vector3f vec );

    float operator[]( int index );

    const float operator[]( int index ) const;

    Vector3f operator+( const Vector3f& p ) const;

    Vector3f operator-( const Vector3f& p ) const;

    Vector3f operator*( float a ) const;

    Vector3f operator/( float a ) const;

    bool operator==( const Vector3f& p );

    bool operator!=( const Vector3f& p );
    Vector3f();
    Vector3f(float _x, float _y, float _z);
    ~Vector3f();
    Vector3f( const Vector3f& p );
private:
    float x;
    float y;
    float z;
};
