#pragma once

class Vector2f {
public:
    void setX( float _x );

    void setY( float _y );

    [[nodiscard]] float getX() const;

    [[nodiscard]] float getY() const;

    void set( const Vector2f& p );
    //operators

    Vector2f& operator=( const Vector2f& p );

    [[nodiscard]] Vector2f normalize() const;

    float& operator[]( int index );

    const float& operator[]( int index ) const;

    Vector2f operator+( const Vector2f& p ) const;

    Vector2f operator-( const Vector2f& p ) const;

    Vector2f operator*( float a ) const;

    Vector2f operator/( float a ) const;

    bool operator==( const Vector2f& p ) const;

    bool operator!=( const Vector2f& p ) const;
    Vector2f();
    Vector2f(float _x, float _y );
    ~Vector2f();
    Vector2f( const Vector2f& p );
private:
    float x{};
    float y{};
};

