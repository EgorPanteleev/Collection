#include <fstream>
#include <vector>
#include <cstdint>

#pragma pack(push, 1) // Ensure structure packing
struct BitmapFileHeader {
    uint16_t type;
    uint32_t size;
    uint16_t reserved1;
    uint16_t reserved2;
    uint32_t offset;
};

struct BitmapInfoHeader {
    uint32_t size;
    int32_t width;
    int32_t height;
    uint16_t planes;
    uint16_t bitCount;
    uint32_t compression;
    uint32_t sizeImage;
    int32_t xPelsPerMeter;
    int32_t yPelsPerMeter;
    uint32_t clrUsed;
    uint32_t clrImportant;
};
#pragma pack(pop)

class Bitmap {
public:
     Bitmap(int width, int height);

     void setPixel(int x, int y, uint8_t red, uint8_t green, uint8_t blue);

     void save(const std::string& filename);

private:
    int width;
    int height;
    std::vector<uint8_t> pixels;
};
