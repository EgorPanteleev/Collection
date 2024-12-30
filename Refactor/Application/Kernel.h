//
// Created by auser on 12/30/24.
//

#ifndef COLLECTION_KERNEL_H
#define COLLECTION_KERNEL_H
#include "hip/hip_runtime.h"
#include <hip/hip_runtime_api.h>
#include "Camera.h"

__global__ void clearMemory( int width, int height, double* memory ) {
    uint32_t x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    uint32_t y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ( x >= width || y >= height ) return;

    uint32_t idx = ( y * width + x ) * 3;
    memory[idx] = 0;
    memory[idx + 1] = 0;
    memory[idx + 2] = 0;
}

__global__ void render( Camera* __restrict__ cam,
                        HittableList* __restrict__ world,
                        unsigned char* __restrict__ colorBuffer,
                        double* __restrict__ memory,
                        double invFrames,
                        hiprandState* __restrict__ states ) {

    uint32_t x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    uint32_t y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if ( x >= cam->imageWidth || y >= cam->imageHeight ) return;
    RGB pixelColor;
    uint32_t idx = y * cam->imageWidth + x;
    hiprandState state = states[idx];
    idx *= 3;

    Ray ray = cam->getRay( x, y, state );
    pixelColor = cam->traceRay(ray, *world, cam->maxDepth, state );

    const Interval<double> intensity( 0, 1 );

    memory[idx] += intensity.clamp( cam->linearToGamma( pixelColor[0] ) ) * 255;
    memory[idx + 1] += intensity.clamp( cam->linearToGamma( pixelColor[1] ) ) * 255;
    memory[idx + 2] += intensity.clamp( cam->linearToGamma( pixelColor[2] ) ) * 255;


    colorBuffer[idx] = memory[idx] * invFrames;
    colorBuffer[idx + 1] = memory[idx + 1] * invFrames;
    colorBuffer[idx + 2] = memory[idx + 2] * invFrames;

}

__global__ void initStates(int width, int height, int seed, hiprandState* states ) {
    uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= width) || (j >= height)) return;
    uint32_t pixel_index = j * width + i;
    hiprand_init(seed, pixel_index, 0, &states[pixel_index] );
}



#endif //COLLECTION_KERNEL_H
