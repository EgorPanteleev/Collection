//
// Created by auser on 7/29/24.
//

#include "Denoiser.h"
#include "OpenImageDenoise/oidn.hpp"
#include <iostream>


void Denoiser::denoise( RGB** data, int w, int h ) {
    //auto& out = mPathTracerBuffers;

    // Initialize OIDN device
    oidn::DeviceRef device = oidn::newDevice(oidn::DeviceType::CPU);
    device.commit();

    // Define buffer size
    const int channels = 3; // ARGB
    // Create the denoising filter
    oidn::FilterRef filter = device.newFilter("RT"); // Use the 'RT' filter for path tracing
    RGB* buffer = new RGB[ w * h ];
    for ( int x = 0; x < w; x++ ) {
        for ( int y = 0; y < h; y++ ) {
            buffer[ y * w + x ] = data[x][y] / 255;
        }
    }
    // Set the input and output buffer (same buffer for in-place denoising)
    filter.setImage("color", buffer, oidn::Format::Float3, w, h, 0, sizeof(float) * channels );
    filter.setImage("output", buffer, oidn::Format::Float3, w, h, 0, sizeof(float) * channels );

    // Set additional parameters if needed
    filter.set("hdr", false); // Assuming the input image is not HDR

    // Commit the filter
    filter.commit();

    // Execute the filter
    filter.execute();

    for ( int x = 0; x < w; x++ ) {
        for ( int y = 0; y < h; y++ ) {
            data[x][y] = buffer[y * w + x] * 255;
        }
    }
}
