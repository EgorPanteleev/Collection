#include "Canvas.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <iostream>

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