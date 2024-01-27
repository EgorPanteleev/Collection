#pragma once

class Vector4f {
public:
    void setX( float _x );

    void setY( float _y );

    void setZ( float _z );

    void setW( float _w );

    [[nodiscard]] float getX() const;

    [[nodiscard]] float getY() const;

    [[nodiscard]] float getZ() const;

    [[nodiscard]] float getW() const;

    void set( const Vector4f& p );
    //operators

    Vector4f& operator=( const Vector4f& p );

    [[nodiscard]] Vector4f normalize() const;

    [[nodiscard]] Vector4f cross( Vector4f vec ) const;

    float& operator[]( int index );

    const float& operator[]( int index ) const;

    Vector4f operator+( const Vector4f& p ) const;

    Vector4f operator-( const Vector4f& p ) const;

    Vector4f operator*( float a ) const;

    Vector4f operator/( float a ) const;

    bool operator==( const Vector4f& p ) const;

    bool operator!=( const Vector4f& p ) const;
    Vector4f();
    Vector4f(float _x, float _y, float _z, float _w);
    ~Vector4f();
    Vector4f( const Vector4f& p );
private:
    float x{};
    float y{};
    float z{};
    float w{};
};



