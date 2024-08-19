
#ifndef COLLECTION_CANVAS_H
#define COLLECTION_CANVAS_H
#include "Color.h"
#include <Kokkos_Core.hpp>
class Canvas {
public:
    Canvas( int _w, int _h );
    Canvas();
    ~Canvas();
    void setColor( int x, int y, const RGB& color );
    [[nodiscard]] RGB getColor( int x, int y ) const;
    void setNormal( int x, int y, const RGB& color );
    [[nodiscard]] RGB getNormal( int x, int y ) const;
    void setAlbedo( int x, int y, const RGB& color );
    [[nodiscard]] RGB getAlbedo( int x, int y ) const;
    [[nodiscard]] int getW() const;
    [[nodiscard]] int getH() const;
    [[nodiscard]] RGB** getColorData() const;
    [[nodiscard]] RGB** getNormalData() const;
    [[nodiscard]] RGB** getAlbedoData() const;
    void saveToPNG( const std::string& fileName ) const;

public:
    RGB** colorData;
    RGB** normalData;
    RGB** albedoData;
    int numX;
    int numY;
};


#endif //COLLECTION_CANVAS_H
