
#ifndef COLLECTION_CANVAS_H
#define COLLECTION_CANVAS_H
#include "Color.h"

class Canvas {
public:
    Canvas( int _w, int _h );
    ~Canvas();
    void setPixel( int x, int y, const RGB& color );
    [[nodiscard]] RGB getPixel( int x, int y ) const;
    [[nodiscard]] int getW() const;
    [[nodiscard]] int getH() const;
private:
    RGB** data;
    int numX;
    int numY;
};


#endif //COLLECTION_CANVAS_H
