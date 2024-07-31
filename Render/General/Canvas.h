
#ifndef COLLECTION_CANVAS_H
#define COLLECTION_CANVAS_H
#include "Color.h"
class Canvas {
public:
    __host__ __device__ Canvas( int _w, int _h );
    __host__ __device__ Canvas();
    __host__ __device__ ~Canvas();
    __host__ __device__ void setPixel( int x, int y, const RGB& color );
    [[nodiscard]] __host__ __device__ RGB getPixel( int x, int y ) const;
    [[nodiscard]] __host__ __device__ int getW() const;
    [[nodiscard]] __host__ __device__ int getH() const;
    [[nodiscard]] __host__ __device__ RGB** getData() const;
    void saveToPNG( const std::string& fileName ) const;

public:
    RGB** data;
    int numX;
    int numY;
};


#endif //COLLECTION_CANVAS_H
