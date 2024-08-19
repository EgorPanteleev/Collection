//
// Created by auser on 7/29/24.
//

#include "Denoiser.h"
#include "OpenImageDenoise/oidn.hpp"

void Denoiser::denoise( RGB** colorData, RGB** normalData, RGB** albedoData, int w, int h ) {
    //auto& out = mPathTracerBuffers;
    // Initialize OIDN device
    oidn::DeviceRef device = oidn::newDevice(oidn::DeviceType::CPU);
    device.commit();

    // Define buffer size
    const int channels = 3; // ARGB
    // Create the denoising filter
    oidn::FilterRef filter = device.newFilter("RT"); // Use the 'RT' filter for path tracing
    RGB* colorBuffer = new RGB[ w * h ];
    Vector3f* normalBuffer = new Vector3f[ w * h ];
    RGB* albedoBuffer = new RGB[ w * h ];
    for ( int x = 0; x < w; x++ ) {
        for ( int y = 0; y < h; y++ ) {
            colorBuffer[ y * w + x ] = colorData[x][y] / 255;
            normalBuffer[ y * w + x ] = normalData[x][y].toNormal();
            albedoBuffer[ y * w + x ] = albedoData[x][y] / 255;
        }
    }

    // Set the input and output buffer (same buffer for in-place denoising)
    filter.setImage("color", colorBuffer, oidn::Format::Float3, w, h );
    filter.setImage("normal", normalBuffer, oidn::Format::Float3, w, h );
    filter.setImage("albedo", albedoBuffer, oidn::Format::Float3, w, h );
    filter.setImage("output", colorBuffer, oidn::Format::Float3, w, h );

    // Set additional parameters if needed
    filter.set("hdr", false); // Assuming the input image is not HDR
    // Commit the filter
    filter.commit();
    // Execute the filter
    filter.execute();

    for ( int x = 0; x < w; x++ ) {
        for ( int y = 0; y < h; y++ ) {
            colorData[x][y] = colorBuffer[y * w + x] * 255;
        }
    }
}
