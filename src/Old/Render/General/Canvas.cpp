#include "Canvas.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <iostream>

Canvas::Canvas( int _w, int _h ) {
    numX = _w;
    numY = _h;
    colorData = new RGB*[numX];
    for ( int i = 0; i < numX; i++ ) {
        colorData[i] = new RGB[numY];
    }
    normalData = new RGB*[numX];
    for ( int i = 0; i < numX; i++ ) {
        normalData[i] = new RGB[numY];
    }
    albedoData = new RGB*[numX];
    for ( int i = 0; i < numX; i++ ) {
        albedoData[i] = new RGB[numY];
    }
}

Canvas::Canvas() {
    numX = 0;
    numY = 0;
}

Canvas::~Canvas() {
//    for ( int i = 0; i < numX; i++ ) {
//        delete[] colorData[i];
//    }
//    delete[] colorData;
//
//    for ( int i = 0; i < numX; i++ ) {
//        delete[] normalData[i];
//    }
//    delete[] normalData;
//
//    for ( int i = 0; i < numX; i++ ) {
//        delete[] albedoData[i];
//    }
//    delete[] albedoData;
}
void Canvas::setColor( int x, int y, const RGB& color ) {
    colorData[x][y] = color;
}

RGB Canvas::getColor( int x, int y ) const {
   return colorData[x][y];
}

void Canvas::setNormal( int x, int y, const RGB& color ) {
    normalData[x][y] = color;
}

RGB Canvas::getNormal( int x, int y ) const {
    return normalData[x][y];
}

void Canvas::setAlbedo( int x, int y, const RGB& color ) {
    albedoData[x][y] = color;
}

RGB Canvas::getAlbedo( int x, int y ) const {
    return albedoData[x][y];
}

int Canvas::getW() const {
    return numX;
}
int Canvas::getH() const {
    return numY;
}

RGB** Canvas::getColorData() const {
    return colorData;
}

RGB** Canvas::getNormalData() const {
    return normalData;
}

RGB** Canvas::getAlbedoData() const {
    return albedoData;
}

void Canvas::saveToPNG( const std::string& fileName ) const {
    // Create an array to store pixel data (RGBA format)
    auto* colors = new unsigned char[numX * numY * 4];
    auto* normals = new unsigned char[numX * numY * 4];
    auto* albedos = new unsigned char[numX * numY * 4];
    for (int y = 0; y < numY; ++y) {
        for (int x = 0; x < numX; ++x) {
            int index = (y * numX + x) * 4;
            RGB color = colorData[x][numY - 1 - y ];
            RGB normal = normalData[x][numY - 1 - y ];
            RGB albedo = albedoData[x][numY - 1 - y ];
            colors[index + 0] = (unsigned char) color.r;
            colors[index + 1] = (unsigned char) color.g;
            colors[index + 2] = (unsigned char) color.b;
            colors[index + 3] = 255;  // Alpha component (opaque)
            normals[index + 0] = (unsigned char) normal.r;
            normals[index + 1] = (unsigned char) normal.g;
            normals[index + 2] = (unsigned char) normal.b;
            normals[index + 3] = 255;  // Alpha component (opaque)
            albedos[index + 0] = (unsigned char) albedo.r;
            albedos[index + 1] = (unsigned char) albedo.g;
            albedos[index + 2] = (unsigned char) albedo.b;
            albedos[index + 3] = 255;  // Alpha component (opaque)
        }
    }

    // Save the image as a PNG file
    if (stbi_write_png( fileName.c_str(), numX, numY, 4, colors, numX * 4))
        std::cout << "Image saved successfully: " << fileName << std::endl;
    else std::cerr << "Failed to save image: " << fileName << std::endl;

    if (stbi_write_png( "Normals.png" , numX, numY, 4, normals, numX * 4))
        std::cout << "Image saved successfully: " << "Normals.png" << std::endl;
    else std::cerr << "Failed to save image: " << "Normals.png" << std::endl;

    if (stbi_write_png( "Albedo.png" , numX, numY, 4, albedos, numX * 4))
        std::cout << "Image saved successfully: " << "Albedo.png" << std::endl;
    else std::cerr << "Failed to save image: " << "Albedo.png" << std::endl;


    // Clean up
    delete[] colors;
    delete[] normals;
    delete[] albedos;
}