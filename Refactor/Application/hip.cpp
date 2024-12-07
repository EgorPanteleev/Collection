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
#include "Vector.h"
#include "Sphere.h"
#include "HittableList.h"
#include "Camera.h"
#include "Vec3.h"
#include "RGB.h"

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
template <typename Type>
void locatedOn( Type* ptr ) {
    hipPointerAttribute_t attributes;
    hipPointerGetAttributes( &attributes, ptr );
    if ( attributes.type == 2 ) {
        printf( "Located on device\n" );
    } else if ( attributes.type == 1 ) {
        printf( "Located on host\n" );
    }
    else if ( attributes.type == 1 ) {
        printf( "Unregistered memory!\n" );
    }
    else {
        printf( "Located somewhere - %d\n", attributes.type );
    }
}


__global__ void
vectoradd_float(float* __restrict__ a, const float* __restrict__ b, const float* __restrict__ c, int width, int height)

{

    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    int i = y * width + x;
    if ( i < (width * height)) {
        a[i] = b[i] + c[i];
    }



}

__global__ void testFloat( float* __restrict__ a )
{
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ( x == 1 && y == 1 ) *a *= 2;
}

__global__ void testVec( Vec3d* __restrict__ vec )
{
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ( x == 1 && y == 1 ) *vec *= 3;
}

__global__ void testVector( Vector<double>* __restrict__ vec )
{
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ( x == 1 && y == 1 ) {
        vec->push_back( 123 );
        for (auto &v: *vec) {
            v *= 4;
        }
    }
}

__global__ void testMetal( Material* __restrict__ mat )
{
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ( x == 1 && y == 1 ) {
        auto metal = (Metal*) mat;
        metal->fuzz *= 5;
        metal->albedo *= { 5, 5, 5 };
    }
}

__global__ void testSphere( Hittable* __restrict__ hittable )
{
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if ( x == 1 && y == 1 ) {
        auto sphere = (Sphere*) hittable;
        sphere->radius *= 6;
        Ray ray( {0,0,0}, {1,1,1,} );
        IntervalD interval( 0, infinity );
        HitRecord hr;
        //if ( sphere->hit( ray, interval, hr ) ) printf("Hitted!\n");
    }
}

__global__ void testVecHittable( Vector<Hittable*>* __restrict__ vec )
{
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ( x == 1 && y == 1 ) {
        auto hittable = (*vec)[0];
        auto sphere = ( Sphere* ) hittable;
        printf( "sphere - rad - %f, orig -  %f, %f, %f\n", sphere->radius, sphere->origin[0], sphere->origin[1], sphere->origin[2] );
        //RGB color = ((Lambertian*)( hittable->material ) )->albedo;
        //printf( "color - %f, %f, %f\n", color.r, color.g, color.b );
        Ray ray( {0,0,0}, {1,1,1,} );
        IntervalD interval( 0, infinity );
        HitRecord hr;
        //if ( hittable->hit( ray, interval, hr ) ) printf("Hitted!\n");
    }
}

__global__ void testWorld( HittableList* __restrict__ world )
{
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    int seed = 1234;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
   // printf( "%f\n", randomDouble( -1, 1, idx, seed ) );
    if ( x == 1 && y == 1 ) {
        auto type = world->spheres[0]->material->type;
        RGB albedo = ( (Lambertian*) ( world->spheres[0]->material ) )->albedo;
        printf( "albedo of 1 material - %f, %f, %f\n", albedo.r, albedo.g, albedo.b );
        printf( "type of 1 material - %d\n", type );
    }
}


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


__global__ void render( Camera* __restrict__ cam, HittableList* __restrict__ world, unsigned char* __restrict__ colorBuffer, int width, int height, hiprandState* states )
{

    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if ( x > width || y >= height ) return;
    const double pixelSamplesScale = 1.0 / cam->samplesPerPixel;
    RGB pixelColor = { 0, 0, 0 };
    int idx = y * width + x;
    hiprandState state = states[idx];
    for ( int s = 0; s < cam->samplesPerPixel; ++s ) {
        idx += s;
        Vec3d pixelCenter = cam->pixel00Loc + (x * cam->pixelDeltaU) + (y * cam->pixelDeltaV);
//        double u = (x + randomDouble(state) - 0.5) / width;
//        double v = (y + randomDouble(state) - 0.5) / height;
        Ray ray = cam->getRay( x, y, state );
        pixelColor += cam->traceRay(ray, *world, cam->maxDepth, state );
    }
    writeColor(colorBuffer, pixelColor * pixelSamplesScale, x, y, cam->imageWidth);


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

    float* hostA;
    float* hostB;
    float* hostC;

    float* deviceA;
    float* deviceB;
    float* deviceC;

    hipDeviceProp_t devProp;
    hipGetDeviceProperties(&devProp, 0);
    cout << " System minor " << devProp.minor << endl;
    cout << " System major " << devProp.major << endl;
    cout << " agent prop name " << devProp.name << endl;



    cout << "hip Device prop succeeded " << endl ;


    int i;
    int errors;

    hostA = (float*)malloc(NUM * sizeof(float));
    hostB = (float*)malloc(NUM * sizeof(float));
    hostC = (float*)malloc(NUM * sizeof(float));

    // initialize the input data
    for (i = 0; i < NUM; i++) {
        hostB[i] = (float)i;
        hostC[i] = (float)i*100.0f;
    }

    HIP_ASSERT(hipMalloc((void**)&deviceA, NUM * sizeof(float)));
    HIP_ASSERT(hipMalloc((void**)&deviceB, NUM * sizeof(float)));
    HIP_ASSERT(hipMalloc((void**)&deviceC, NUM * sizeof(float)));

    HIP_ASSERT(hipMemcpy(deviceB, hostB, NUM*sizeof(float), hipMemcpyHostToDevice));
    HIP_ASSERT(hipMemcpy(deviceC, hostC, NUM*sizeof(float), hipMemcpyHostToDevice));


    hipLaunchKernelGGL(vectoradd_float,
                       dim3(WIDTH/THREADS_PER_BLOCK_X, HEIGHT/THREADS_PER_BLOCK_Y),
                       dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y),
                       0, 0,
                       deviceA ,deviceB ,deviceC ,WIDTH ,HEIGHT);


    HIP_ASSERT(hipMemcpy(hostA, deviceA, NUM*sizeof(float), hipMemcpyDeviceToHost));

    // verify the results
    errors = 0;
    for (i = 0; i < NUM; i++) {
       // std::cout << hostB[i] << " + " << hostC[i] << " = " << hostA[i] << std::endl;
        if (hostA[i] != (hostB[i] + hostC[i])) {
            errors++;
        }
    }
    if (errors!=0) {
        printf("FAILED: %d errors\n",errors);
    } else {
        printf ("PASSED!\n");
    }

    HIP_ASSERT(hipFree(deviceA));
    HIP_ASSERT(hipFree(deviceB));
    HIP_ASSERT(hipFree(deviceC));

    free(hostA);
    free(hostB);
    free(hostC);

    //hipResetDefaultAccelerator();

//    Vec3d* vec1 = new Vec3d( 0, 1, 2 );
//    Vec3d* deviceVec1;
//    HIP::copyToDevice( vec1, deviceVec1 );
//    Vec3d* res = new Vec3d( 0, 1, 1 );
//    HIP::copyToHost( res, deviceVec1 );
//
//    std::cout << "result - " << *res;
//
//    float* f1 = (float*)malloc(sizeof(float));
//    *f1 = 1;
//    float* deviceFloat;
//    HIP::copyToDevice( f1, deviceFloat );
////    HIP_ASSERT(hipMalloc(&deviceFloat, sizeof(float)));
////    HIP_ASSERT(hipMemcpy(deviceFloat, f1, sizeof(float), hipMemcpyHostToDevice));
//    float* f2 = (float*)malloc(sizeof(float));
//    *f2 = 0;
//
//    hipLaunchKernelGGL(add,
//                       dim3(WIDTH/THREADS_PER_BLOCK_X, HEIGHT/THREADS_PER_BLOCK_Y),
//                       dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y),
//                       0, 0,
//                       deviceFloat );
//
//
//    HIP::copyToHost( f2, deviceFloat );
//    std::cout << "result float - " << *f2 << std::endl;


//    Vec3d v1( 0, 1, 34 );
//    Vec3d* v1Dev;
//    HIP::copyToDevice( &v1, v1Dev );
//    Vec3d v2;
//    HIP::copyToHost( &v2, v1Dev );
//
//    std::cout << "result finis - " << v2 << std::endl;

//    Vector<double> vector1;
//    vector1.reserve( 10000 );
//
//    vector1.push_back( 1 );
//    vector1.push_back( 2 );
//    vector1.push_back( 3 );
//    vector1.push_back( 105 );
//
//    Vector<double>* vectorDevice;
//
//    vector1.copyToDevice( vectorDevice );
//
//    Vector<double> vector2;
//    vectorDevice->copyToHost( &vector2 );
//
//    std::cout << vector2[0] << " " << vector2[1] << " " << vector2[2] << " " << vector2[3] << " " <<
//    vector2.capacity() << " " << vector2.size() << std::endl;

//    HittableList world;
//    Material* ground = new Lambertian( { 0.8, 0.8, 0.0 } );
//    Material* center = new Lambertian( { 0.1, 0.2, 0.5 } );
//    Material* left = new Dielectric( 1.5 );
//    Material* bubble = new Dielectric( 1.0 / 1.5 );
//    Material* right = new Metal( { 0.8, 0.6, 0.2 }, 1.0 );
//    world.add( new Sphere( 100, { 0, -100.5, -1 }, ground ) );
//    world.add( new Sphere( 0.5, { 0, 0, -1.2 }, center ) );
//    world.add( new Sphere( 0.5, { -1, 0, -1 }, left ) );
//    world.add( new Sphere( 0.4, { -1, 0, -1 }, bubble ) );
//    world.add( new Sphere( 0.5, { 1, 0, -1 }, right ) );
//
//    HittableList* worldDevice;
//
//    world.copyToDevice( worldDevice );

    int cntErrors = 0;
// test float

    float f1 = 123;
    auto* devFloat = HIP::allocateOnDevice<float>();
    HIP::copyToDevice( &f1, devFloat );

    hipLaunchKernelGGL(testFloat,
                       dim3(WIDTH/THREADS_PER_BLOCK_X, HEIGHT/THREADS_PER_BLOCK_Y),
                       dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y),
                       0, 0,
                       devFloat );
    float hostFloat = -1;
    HIP::copyToHost( &hostFloat, devFloat );
    HIP::deallocateOnDevice( devFloat );
    if ( hostFloat != 246 ) ++cntErrors;


//test Vec3d

    Vec3d vec1 = { 1, 2, 3 };
    auto* vecDevice = HIP::allocateOnDevice<Vec3d>();
    HIP::copyToDevice( &vec1, vecDevice );

    hipLaunchKernelGGL(testVec,
                       dim3(WIDTH/THREADS_PER_BLOCK_X, HEIGHT/THREADS_PER_BLOCK_Y),
                       dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y),
                       0, 0,
                       vecDevice );

    Vec3d hostVec = { -1, -1, -1 };
    HIP::copyToHost( &hostVec, vecDevice );
    HIP::deallocateOnDevice( vecDevice );

    if ( hostVec != Vec3d( 3, 6, 9 ) ) ++cntErrors;

//test vector

    Vector<double> vector;
    vector.reserve( 10000 );

    vector.push_back( 1 );
    vector.push_back( 2 );
    vector.push_back( 3 );
    vector.push_back( 105 );

    Vector<double>* vectorDevice = vector.copyToDevice();

    hipLaunchKernelGGL(testVector,
                       dim3(WIDTH/THREADS_PER_BLOCK_X, HEIGHT/THREADS_PER_BLOCK_Y),
                       dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y),
                       0, 0,
                       vectorDevice );


    Vector<double> hostVector = *vectorDevice->copyToHost();


    bool vecEqual = hostVector[0] == 4 && hostVector[1] == 8 && hostVector[2] == 12 && hostVector[3] == 420 && hostVector[4] == 492 && hostVector.size() == 5 && hostVector.capacity() == 10000;

    std::cout << hostVector[0] << " " << hostVector[1] << " " << hostVector[2] << " " << hostVector[3] << " " << hostVector.size() << " " << hostVector.capacity() << std::endl;

    if ( !vecEqual ) ++cntErrors;
//test material
    Material* m1 = new Metal( { 1, 1, 1 }, 0.15 );
    Material* deviceMaterial = m1->copyToDevice();

    hipLaunchKernelGGL(testMetal,
                       dim3(WIDTH/THREADS_PER_BLOCK_X, HEIGHT/THREADS_PER_BLOCK_Y),
                       dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y),
                       0, 0,
                       deviceMaterial );


    Material* m2 =  deviceMaterial->copyToHost();



    auto m3 = (Metal*) m2;

    bool metalEqual = m3->fuzz == 0.75 && m3->albedo.r == 5 && m3->albedo.g == 5 && m3->albedo.b == 5;

    std::cout << m3->fuzz << " " << m3->albedo.r << " " << m3->albedo.g << " " << m3->albedo.b << std::endl;

    if ( !metalEqual ) ++cntErrors;
//test Hittable
    Sphere* obj1 = new Sphere( 111, { 2, 2, 2 }, ( (Metal*) m1 )  );
    Sphere* devObj = obj1->copyToDevice( );

    hipLaunchKernelGGL(testSphere,
                       dim3(WIDTH/THREADS_PER_BLOCK_X, HEIGHT/THREADS_PER_BLOCK_Y),
                       dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y),
                       0, 0,
                       devObj );


    Sphere* obj2 = devObj->copyToHost();
    auto objSph = (Sphere*) obj2;

    if ( objSph->radius != 666 ) ++cntErrors;

   // std::cout << objSph->radius << " " << objSph->origin << " " << ( (Metal*) objSph->material )->fuzz << " " << ( (Metal*) objSph->material )->albedo.r << " " << ( (Metal*) objSph->material )->albedo.g << " " << ( (Metal*) objSph->material )->albedo.b << std::endl;

    //data

    Lambertian* ground = new Lambertian( { 0.8, 0.8, 0.0 } );
    Lambertian* center = new Lambertian( { 0.1, 0.2, 0.5 } );
    Dielectric* left = new Dielectric( 1.5 );
    Dielectric* bubble = new Dielectric( 1.0 / 1.5 );
    Metal* right = new Metal( { 0.8, 0.6, 0.2 }, 1.0 );


    //test vector hittable

    Vector<Sphere*> wrld;

    wrld.push_back( new Sphere( 100, { 0, -100.5, -1 }, ground ) );
    wrld.push_back( new Sphere( 0.5, { 0, 0, -1.2 }, center ) );

    auto wrldDevice = wrld.copyToDevice();


    //locatedOn( wrldDevice );


//    hipLaunchKernelGGL(testVecHittable,
//                       dim3(WIDTH/THREADS_PER_BLOCK_X, HEIGHT/THREADS_PER_BLOCK_Y),
//                       dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y),
//                       0, 0,
//                       wrldDevice );

    auto wrldHost = wrldDevice->copyToHost();

    locatedOn( wrldHost );

    HittableList world;

    world.add( new Sphere( 100, { 0, -100.5, -1 }, ground ) );
    world.add( new Sphere( 0.5, { 0, 0, -1.2 }, center ) );
    world.add( new Sphere( 0.5, { -1, 0, -1 }, left ) );
    world.add( new Sphere( 0.4, { -1, 0, -1 }, bubble ) );
    world.add( new Sphere( 0.5, { 1, 0, -1 }, right ) );
////
    auto worldDevice = world.copyToDevice();

    hipLaunchKernelGGL(testWorld,
                       dim3(WIDTH/THREADS_PER_BLOCK_X, HEIGHT/THREADS_PER_BLOCK_Y),
                       dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y),
                       0, 0,
                       worldDevice );

    //auto worldHost = worldDevice->copyToHost();

    //std::cout << "TEST - " << ( ( Sphere*) worldHost->spheres[4] )->radius << std::endl;

    Camera cam;
    cam.aspectRatio = 16.0 / 10.0;
    cam.imageWidth = 800;
    cam.samplesPerPixel = 100;
    cam.maxDepth = 30;
    cam.vFOV = 30;

    cam.lookFrom = { 0, 0, 1 };
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

//    hipLaunchKernelGGL(render,
//                       dim3(WIDTH/THREADS_PER_BLOCK_X, HEIGHT/THREADS_PER_BLOCK_Y),
//                       dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y),
//                       0, 0,
//                       deviceCamera, worldDevice, deviceColorBuffer, cam.imageWidth, cam.imageHeight );

    render<<< dim3(WIDTH/THREADS_PER_BLOCK_X, HEIGHT/THREADS_PER_BLOCK_Y), dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y)>>>(deviceCamera, worldDevice, deviceColorBuffer, cam.imageWidth, cam.imageHeight, states );

    hipDeviceSynchronize();

    timer.end();

    auto colorBuffer = new unsigned char[cam.imageWidth * cam.imageHeight * 4];

    HIP::copyToHost( colorBuffer, deviceColorBuffer, cam.imageWidth * cam.imageHeight * 4 );

    std::cout << "RayTracer works "<< timer.get() << " seconds" << std::endl;

    saveToPNG( "outHIP.png", colorBuffer, cam.imageWidth, cam.imageHeight );


    //std::cout << ((Metal*)((Sphere*)(*worldHost).objects[4])->material)->fuzz << std::endl;

//    Vector<Hittable*> world;
//    world.push_back( new Sphere( 100, { 0, -100.5, -1 }, ground ) );
//    world.push_back( new Sphere( 0.5, { 0, 0, -1.2 }, center ) );
//    world.push_back( new Sphere( 0.5, { -1, 0, -1 }, left ) );
//    world.push_back( new Sphere( 0.4, { -1, 0, -1 }, bubble ) );
//    world.push_back( new Sphere( 0.5, { 1, 0, -1 }, right ) );
//
//    auto worldDevice = world.copyToDevice<true>();
//
//    auto worldHost = worldDevice->copyToHost<true>();
//
//    std::cout << ((Metal*)((Sphere*)(*worldHost)[4])->material)->fuzz << std::endl;

    if ( cntErrors ) std::cout << "Failed!" << std::endl;
    else std::cout << "Passed!" << std::endl;
    return errors;
}

//// TODO сделать обертку из класс для аллоцирования на девайся
/// хуячить крутые тесты

#else
#include <iostream>

int main() {
    std::cout << "HIP is not enabled" << std::endl;
}


#endif