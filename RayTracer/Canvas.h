
#ifndef COLLECTION_CANVAS_H
#define COLLECTION_CANVAS_H
#include "Color.h"

class Canvas {
public:
    Canvas( const int _w, const int _h );
    ~Canvas();
    void setPixel( int x, int y, RGB color );
    RGB getPixel( int x, int y ) const;
    int getW() const;
    int getH() const;
private:
    RGB** data;
    int numX;
    int numY;
};


#endif //COLLECTION_CANVAS_H
