/*
Copyright (c) 2015-2016 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#ifdef HIP_ENABLED

#include <assert.h>
#include <stdio.h>
#include <algorithm>
#include <stdlib.h>
#include<iostream>
#include "hip/hip_runtime.h"
#include <hip/hip_runtime_api.h>
#include "Vector.h"
#include "Sphere.h"
#include "HittableList.h"
#include "Camera.h"
#include "Vec3.h"
#include "RGB.h"
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <hip/hip_gl_interop.h>
#ifdef NDEBUG
#define HIP_ASSERT(x) x
#else
#define HIP_ASSERT(x) (assert((x)==hipSuccess))
#endif

#include "Timer.h"
#define WIDTH     1024
#define HEIGHT    1024

#define NUM       (WIDTH*HEIGHT)

#define THREADS_PER_BLOCK_X 16
#define THREADS_PER_BLOCK_Y 16

#define THREADS_PER_BLOCK_Z  1

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


HOST_DEVICE double linearToGamma( double linear ) {
    if ( linear > 0 ) return std::sqrt( linear );
    return 0;
}

HOST_DEVICE void writeColor( unsigned char* colorBuffer, const RGB& color, int i, int j, int imageWidth ) {
    int index = (j * imageWidth + i) * 4;
    const Interval<double> intensity( 0, 0.999 );
    colorBuffer[index + 0] = (unsigned char) ( intensity.clamp( linearToGamma( color.r ) ) * 256 );
    colorBuffer[index + 1] = (unsigned char) ( intensity.clamp( linearToGamma( color.g ) ) * 256 );
    colorBuffer[index + 2] = (unsigned char) ( intensity.clamp( linearToGamma( color.b ) ) * 256 );
    colorBuffer[index + 3] = 255;
}


__global__ void render( hipSurfaceObject_t surface, int frame, Camera* __restrict__ cam, HittableList* __restrict__ world, unsigned char* __restrict__ colorBuffer, int width, int height, hiprandState* states )
{

    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if ( x > width || y >= height ) return;
    const double pixelSamplesScale = 1.0 / cam->samplesPerPixel;
    RGB pixelColor = { 0, 0, 0 };
    int idx = y * width + x;
    hiprandState state = states[idx];

    Vec3d pixelCenter = cam->pixel00Loc + (x * cam->pixelDeltaU) + (y * cam->pixelDeltaV);
    //Ray ray = cam->getRay( x, y, state );
    //pixelColor += cam->traceRay(ray, *world, cam->maxDepth, state );

    uchar4 pixel;
    pixel.x = (x + frame) % 256; // Red
    pixel.y = (y + frame) % 256; // Green
    pixel.z = (x + y + frame) % 256; // Blue
    pixel.w = 255; // Alpha

    surf2Dwrite(pixel, surface, x * sizeof(uchar4), y);
    //writeColor(colorBuffer, pixelColor * pixelSamplesScale, x, y, cam->imageWidth);


}

__global__ void render_init(int max_x, int max_y, hiprandState* states ) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j*max_x + i;
    //Each thread gets same seed, a different sequence number, no offset
    hiprand_init(1984, pixel_index, 0, &states[pixel_index] );
}



using namespace std;
#include "Vec3.h"
#include "SystemUtils.h"
#include "Hittable.h"


void saveToPNG( const std::string& fileName, unsigned char* colorBuffer, int imageWidth, int imageHeight ) {
    if (stbi_write_png( fileName.c_str(), imageWidth, imageHeight, 4, colorBuffer, imageWidth * 4))
        std::cout << "Image saved successfully: " << fileName << std::endl;
    else std::cerr << "Failed to save image: " << fileName << std::endl;
}

int main() {

    Lambertian* ground = new Lambertian( { 0.8, 0.8, 0.0 } );
    Lambertian* center = new Lambertian( { 0.1, 0.2, 0.5 } );
    Dielectric* left = new Dielectric( 1.5 );
    Dielectric* bubble = new Dielectric( 1.0 / 1.5 );
    Metal* right = new Metal( { 0.8, 0.6, 0.2 }, 1.0 );

    HittableList world;

    world.add( new Sphere( 100, { 0, -100.5, -1 }, ground ) );
    world.add( new Sphere( 0.5, { 0, 0, -1.2 }, center ) );
    world.add( new Sphere( 0.5, { -1, 0, -1 }, left ) );
    world.add( new Sphere( 0.4, { -1, 0, -1 }, bubble ) );
    world.add( new Sphere( 0.5, { 1, 0, -1 }, right ) );

    auto worldDevice = world.copyToDevice();


    Camera cam;
    cam.aspectRatio = 16.0 / 10.0;
    cam.imageWidth = 800;
    cam.samplesPerPixel = 1;
    cam.maxDepth = 30;
    cam.vFOV = 30;

    cam.lookFrom = { -2, 2, 1 };
    cam.lookAt = { 0, 0, -1 };
    cam.up = { 0, 1, 0 };

    cam.init();


    hiprandState *states;
    HIP_ASSERT(hipMalloc((void **)&states, cam.imageWidth * cam.imageHeight * sizeof(hiprandState)));
    render_init<<< dim3(WIDTH/THREADS_PER_BLOCK_X, HEIGHT/THREADS_PER_BLOCK_Y), dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y)>>>(cam.imageWidth, cam.imageHeight, states);

    auto deviceCamera = HIP::allocateOnDevice<Camera>();

    HIP::copyToDevice( &cam, deviceCamera );

    auto deviceColorBuffer = HIP::allocateOnDevice<unsigned char>(cam.imageWidth * cam.imageHeight * 4);

    Timer timer;

    timer.start();

//    render<<< dim3(WIDTH/THREADS_PER_BLOCK_X, HEIGHT/THREADS_PER_BLOCK_Y), dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y)>>>
//    (deviceCamera, worldDevice, deviceColorBuffer, cam.imageWidth, cam.imageHeight, states );

    hipDeviceSynchronize();

    timer.end();

    auto colorBuffer = new unsigned char[cam.imageWidth * cam.imageHeight * 4];

    HIP::copyToHost( colorBuffer, deviceColorBuffer, cam.imageWidth * cam.imageHeight * 4 );

    std::cout << "RayTracer works "<< timer.get() << " seconds" << std::endl;

    saveToPNG( "outHIP.png", colorBuffer, cam.imageWidth, cam.imageHeight );


    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW\n";
        return -1;
    }

    int width = cam.imageWidth;
    int height = cam.imageHeight;

    // Создание окна
    GLFWwindow* window = glfwCreateWindow(width, height, "HIP OpenGL Interop", NULL, NULL);
    if (!window) {
        std::cerr << "Failed to create GLFW window\n";
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);


    // Инициализация GLEW
    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW\n";
        return -1;
    }

    GLuint texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glBindTexture(GL_TEXTURE_2D, 0);

    // --- HIP Interop: Регистрация текстуры ---
    hipGraphicsResource* cudaResource;
    hipGraphicsGLRegisterImage(&cudaResource, texture, GL_TEXTURE_2D, hipGraphicsRegisterFlagsWriteDiscard);

    int frame = 0;
    while (!glfwWindowShouldClose(window)) {
        // --- HIP: Захват ресурса ---
        hipGraphicsMapResources(1, &cudaResource, 0);

        hipArray* array;
        hipGraphicsSubResourceGetMappedArray(&array, cudaResource, 0, 0);

        // Map array into a surface for kernel writing
        hipSurfaceObject_t surfaceObj;
        hipCreateSurfaceObject(&surfaceObj, array);

        // Запуск HIP ядра
        dim3 blockSize(16, 16);
        dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
        render<<< gridSize, blockSize>>>
        ( surface, frame++, deviceCamera, worldDevice, deviceColorBuffer, cam.imageWidth, cam.imageHeight, states );
        //hipLaunchKernelGGL(render, gridSize, frame++, blockSize, 0, 0, array, width, height );

        // Освобождение ресурса
        hipGraphicsUnmapResources(1, &cudaResource, 0);

        // Рендеринг текстуры на экран
        glClear(GL_COLOR_BUFFER_BIT);
        glBindTexture(GL_TEXTURE_2D, texture);
        glBegin(GL_QUADS);
        glTexCoord2f(0.0f, 0.0f); glVertex2f(-1.0f, -1.0f);
        glTexCoord2f(1.0f, 0.0f); glVertex2f(1.0f, -1.0f);
        glTexCoord2f(1.0f, 1.0f); glVertex2f(1.0f, 1.0f);
        glTexCoord2f(0.0f, 1.0f); glVertex2f(-1.0f, 1.0f);
        glEnd();

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // Очистка
    hipGraphicsUnregisterResource(cudaResource);
    glDeleteTextures(1, &texture);
    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}


#endif