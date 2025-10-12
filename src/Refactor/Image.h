//
// Created by auser on 2/5/25.
//

#ifndef COLLECTION_IMAGE_H
#define COLLECTION_IMAGE_H

#include <stb_image.h>
#include "Vec3.h"
#include "SystemUtils.h"

class Image {
public:
    Image(): data( nullptr ), width( 0 ), height( 0 ), channels( 0 ) {}
    Image( const std::string& path ) {
        load( path );
    }
    ~Image() {
        delete data;
    }
    void load( const std::string& path ) {
        data = stbi_load( path.c_str(), &width, &height, &channels, 0);
        if (!data) std::cerr << "Error loading image: " << path << std::endl;
    }
    [[nodiscard]] HOST_DEVICE Vec3i getPixelColor( const uint x, const uint y ) const {
        uint idx = ( y * width + x ) * channels;
        return { data[idx], data[idx + 1], data[idx + 2] };
    }
    [[nodiscard]] HOST_DEVICE bool empty() const { return data == nullptr; }
    [[nodiscard]] HOST_DEVICE uint size() const { return width * height * channels; }

#if HIP_ENABLED
    HOST Image* copyToDevice() {
        auto device = HIP::allocateOnDevice<Image>();
        HIP::copyToDevice( this, device );
        device->data = HIP::allocateOnDevice<unsigned char>( size() );
        HIP::copyToDevice( data, device->data, size() );
        return device;
    }

    HOST Image* copyToHost() {
        auto host = new Image();
        HIP::copyToHost( host, this );
        HIP::copyToHost( host->data, data, size() );
        return host;
    }

    HOST void deallocateOnDevice() {
        HIP::deallocateOnDevice<unsigned char>( data );
        HIP::deallocateOnDevice<Image>( this );
    }

#endif

    int width;
    int height;
    int channels;
    unsigned char* data;
public:
};


#endif //COLLECTION_IMAGE_H
