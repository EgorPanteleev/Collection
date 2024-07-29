#include "Canvas.h"

Canvas::Canvas( int _w, int _h ) {
    numX = _w;
    numY = _h;
    data = new RGB*[numX];
    for ( int i = 0; i < numX; i++ ) {
        data[i] = new RGB[numY];
    }
}

Canvas::Canvas() {
    numX = 0;
    numY = 0;
}

Canvas::~Canvas() {
    //delete[] data;
}
void Canvas::setPixel( int x, int y, const RGB& color ) {
    data[x][y] = color;
}

RGB Canvas::getPixel( int x, int y ) const {
   return data[x][y];
}

int Canvas::getW() const {
    return numX;
}
int Canvas::getH() const {
    return numY;
}

[[nodiscard]] RGB** Canvas::getData() const {
    return data;
}