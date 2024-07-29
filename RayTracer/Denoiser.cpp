//
// Created by auser on 7/29/24.
//

#include "Denoiser.h"
#include "OpenImageDenoise/oidn.hpp"
#include <iostream>

Denoiser::Denoiser( Canvas* _canvas ) {
    canvas = _canvas;
}

void Denoiser::denoise() {
    //auto& out = mPathTracerBuffers;

    // Initialize OIDN device
    oidn::DeviceRef device = oidn::newDevice(oidn::DeviceType::CPU);
    device.commit();

    // Define buffer size
    const int channels = 3; // ARGB
    // Create the denoising filter
    oidn::FilterRef filter = device.newFilter("RT"); // Use the 'RT' filter for path tracing
    RGB* data = new RGB[canvas->getW() * canvas->getH()];
    for ( int x = 0; x < canvas->getW(); x++ ) {
        for ( int y = 0; y < canvas->getH(); y++ ) {
            data[ y * canvas->getW() + x ] = canvas->getPixel( x, y ) / 255;
        }
    }
    // Set the input and output buffer (same buffer for in-place denoising)
    filter.setImage("color", data, oidn::Format::Float3, canvas->getW(), canvas->getH(), 0, sizeof(float) * channels );
    filter.setImage("output", data, oidn::Format::Float3, canvas->getW(), canvas->getH(), 0, sizeof(float) * channels );

    // Set additional parameters if needed
    filter.set("hdr", false); // Assuming the input image is not HDR

    // Commit the filter
    filter.commit();

    // Execute the filter
    filter.execute();

    for ( int x = 0; x < canvas->getW(); x++ ) {
        for ( int y = 0; y < canvas->getH(); y++ ) {
            canvas->setPixel( x, y, data[y * canvas->getW() + x] * 255 );
        }
    }
}
