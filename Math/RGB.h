//
// Created by auser on 10/20/24.
//

#ifndef MATH_RGB_H
#define MATH_RGB_H
#undef RGB
#define BLACK RGB( 20, 20, 20 )
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
#include <iostream>

class RGB {
public:
    RGB();
    RGB( double r, double g, double b);
    RGB( const RGB& other );
    ~RGB();

    RGB& operator=( const RGB& other );

    RGB& operator+=( const RGB& other );
    RGB operator+( const RGB& other ) const;

    RGB& operator*=( const RGB& other );
    RGB operator*( const RGB& other ) const;

    RGB& operator/=( const RGB& other );
    RGB operator/( const RGB& other ) const;

    bool operator==( const RGB& other ) const;
    bool operator!=( const RGB& other ) const;
    void scaleTo( double value );

public:
    double r;
    double g;
    double b;
};

RGB operator+( const RGB& col, const double& d );
RGB operator+( const double& d, const RGB& col );
RGB operator*( const RGB& col, const double& d );
RGB operator*( const double& d, const RGB& col );
RGB operator/( const RGB& col, const double& d );
RGB operator/( const double& d, const RGB& col );



#endif //MATH_RGB_H
