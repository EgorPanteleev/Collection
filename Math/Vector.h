#ifndef COLLECTION_Vector3f_H
#define COLLECTION_Vector3f_H


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

    Vector3f operator+( const Vector3f& p ) const;

    Vector3f operator-( const Vector3f& p ) const;

    Vector3f operator*( float a ) const;

    Vector3f operator/( float a ) const;

    bool operator==( const Vector3f& p );

    bool operator!=( const Vector3f& p );
    Vector3f(): x(0), y(0), z(0){ }
    Vector3f(float _x, float _y, float _z): x(_x), y(_y), z(_z){ }
    ~Vector3f();
    Vector3f( const Vector3f& p );
private:
    float x;
    float y;
    float z;
};


#endif //COLLECTION_Vector3f_H
