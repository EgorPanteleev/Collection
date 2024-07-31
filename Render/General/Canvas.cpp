#include "Canvas.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <iostream>

__host__ __device__ Canvas::Canvas( int _w, int _h ) {
    numX = _w;
    numY = _h;
    data = new RGB*[numX];
    for ( int i = 0; i < numX; i++ ) {
        data[i] = new RGB[numY];
    }
}

__host__ __device__ Canvas::Canvas() {
    numX = 0;
    numY = 0;
}

__host__ __device__ Canvas::~Canvas() {
    //delete[] data;
}
__host__ __device__ void Canvas::setPixel( int x, int y, const RGB& color ) {
    data[x][y] = color;
}

__host__ __device__ RGB Canvas::getPixel( int x, int y ) const {
   return data[x][y];
}

__host__ __device__ int Canvas::getW() const {
    return numX;
}
__host__ __device__ int Canvas::getH() const {
    return numY;
}

[[nodiscard]] __host__ __device__ RGB** Canvas::getData() const {
    return data;
}

void Canvas::saveToPNG( const std::string& fileName ) const {
    // Create an array to store pixel data (RGBA format)
    auto* image = new unsigned char[numX * numY * 4];
    for (int y = 0; y < numY; ++y) {
        for (int x = 0; x < numX; ++x) {
            int index = (y * numX + x) * 4;
            RGB color = data[x][numY - 1 - y ];
            image[index + 0] = (unsigned char) color.r;
            image[index + 1] = (unsigned char) color.g;
            image[index + 2] = (unsigned char) color.b;
            image[index + 3] = 255;  // Alpha component (opaque)
        }
    }

    // Save the image as a PNG file
    if (stbi_write_png( fileName.c_str(), numX, numY, 4, image, numX * 4))
        std::cout << "Image saved successfully: " << fileName << std::endl;
    else std::cerr << "Failed to save image: " << fileName << std::endl;

    // Clean up
    delete[] image;
}