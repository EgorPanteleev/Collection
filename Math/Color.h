
#ifndef COLLECTION_COLOR_H
#define COLLECTION_COLOR_H
#undef RGB
#define BLACK RGB( 0, 0, 0 )
#define WHITE RGB( 255, 255, 255 )
#define GRAY RGB( 210, 210, 210 )
#define RED RGB( 255, 0, 0 )
#define GREEN RGB( 0, 255, 0 )
#define BLUE RGB( 0, 0, 255 )
#define YELLOW RGB( 255, 255, 0 )
#define BROWN RGB( 150, 75, 0 )
#define PINK RGB( 255,105,180 )
#define DARK_BLUE RGB(65,105,225)
#define CYAN RGB( 0, 255, 255)
#include "Vector3f.h"

class RGB {
public:
    void set( float _r, float _g, float _b );
    RGB operator+( const RGB& color ) const;
    RGB operator*( float a) const;
    RGB operator/( float a) const;
    RGB operator*( const RGB& p ) const;
    RGB operator/( const RGB& p ) const;
    bool operator==( const RGB& color ) const;
    void scaleTo( float value );
    [[nodiscard]] Vector3f toNormal() const;
    RGB();
    RGB( float _r, float _g, float _b);
    ~RGB();
public:
    float r;
    float g;
    float b;
};


#endif //COLLECTION_COLOR_H
