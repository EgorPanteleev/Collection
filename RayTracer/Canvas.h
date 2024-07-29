
#ifndef COLLECTION_CANVAS_H
#define COLLECTION_CANVAS_H
#include "Color.h"
#include <Kokkos_Core.hpp>
class Canvas {
public:
    Canvas( int _w, int _h );
    Canvas();
    ~Canvas();
    void setPixel( int x, int y, const RGB& color );
    [[nodiscard]] RGB getPixel( int x, int y ) const;
    [[nodiscard]] int getW() const;
    [[nodiscard]] int getH() const;
    [[nodiscard]] RGB** getData() const;
public:
    RGB** data;
    int numX;
    int numY;
};


#endif //COLLECTION_CANVAS_H
