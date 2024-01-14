#include "Image.h"

Bitmap::Bitmap(int width, int height) : width(width), height(height) {
    pixels.resize(width * height * 3, 0);
}

void Bitmap::setPixel(int x, int y, uint8_t red, uint8_t green, uint8_t blue) {
    int index = (y * width + x) * 3;
    pixels[index] = blue;
    pixels[index + 1] = green;
    pixels[index + 2] = red;
}

void Bitmap::save(const std::string& filename) {
    std::ofstream bmpFile(filename, std::ios::binary);

    BitmapFileHeader fileHeader{
            .type = 0x4D42, // BM
            .size = sizeof(BitmapFileHeader) + sizeof(BitmapInfoHeader) + width * height * 3,
            .reserved1 = 0,
            .reserved2 = 0,
            .offset = sizeof(BitmapFileHeader) + sizeof(BitmapInfoHeader)
    };
    bmpFile.write(reinterpret_cast<char*>(&fileHeader), sizeof(BitmapFileHeader));

    BitmapInfoHeader infoHeader{
            .size = sizeof(BitmapInfoHeader),
            .width = width,
            .height = height,
            .planes = 1,
            .bitCount = 24,
            .compression = 0,
            .sizeImage = 0,
            .xPelsPerMeter = 0,
            .yPelsPerMeter = 0,
            .clrUsed = 0,
            .clrImportant = 0
    };
    bmpFile.write(reinterpret_cast<char*>(&infoHeader), sizeof(BitmapInfoHeader));

    bmpFile.write(reinterpret_cast<char*>(pixels.data()), pixels.size());

    bmpFile.close();
}

