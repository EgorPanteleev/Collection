#include <iostream>
#include "RayTracer.h"
#include "Scene.h"
#include "Camera.h"
#include "Sphere.h"
#include "Image.h"
#include <time.h>

int main() {
    Camera* cam = new Camera( Vector3f(0,0,0), Vector3f(0,0,1), 1000,1000,1000 );
    Scene* scene = new Scene();
    RayTracer rayt( cam, scene );
    std::vector<Shape*> shapes;
    std::vector<Material> materials;
    std::vector<Object*> objects;
    std::vector<Light*> ligths;
    shapes.push_back(new Sphere(50, Vector3f(120, 120, 300)));
    materials.emplace_back( RGB( 255, 0, 0), 10 , 0 );

    shapes.push_back(new Sphere(20, Vector3f(40, 40, 400)));
    materials.emplace_back( RGB( 0, 255, 0), 10 , 0 );

    shapes.push_back(new Sphere(30, Vector3f(-20, -20, 1000)));
    materials.emplace_back( RGB( 0, 0, 255), 10 , 0 );

    shapes.push_back(new Sphere(1500, Vector3f(0, 0, 3000)));
    materials.emplace_back( RGB( 255, 255, 0), 10 , 1 );

//    spheres.push_back(new Sphere(20, Vector3f(0, 20, 100), RGB(255,0 ,0 )));
//    spheres.push_back(new Sphere(10, Vector3f(40, 20, 100), RGB(0, 255, 0)));
//    spheres.push_back(new Sphere(30, Vector3f(-20, -20, 1000), RGB(0, 0, 255)));
//    spheres.push_back(new Sphere(1500, Vector3f(0, 0, 3000), RGB(255, 255, 0)));
    for ( int i = 0; i < shapes.size(); ++i ) {
        scene->objects.push_back( new Object( shapes[i], materials[i] ) );
    }

    ligths.push_back( new Light( Vector3f(100,120,20), 0.5));
    ligths.push_back( new Light( Vector3f(-300,0,-200), 0.4));
    ligths.push_back( new Light( Vector3f(300,0,700), 0.4));
    for ( auto l: ligths ) {
        scene->lights.push_back( l );
    }
    clock_t start = clock();
    rayt.traceAllRays();
    clock_t end = clock();
    double seconds = (double)(end - start) / CLOCKS_PER_SEC;
    printf("The time: %f seconds\n", seconds); // best - 2.029000
    Bitmap bmp(rayt.getCanvas()->getW(), rayt.getCanvas()->getH());
    for (int x = 0; x < rayt.getCanvas()->getW(); ++x) {
        for (int y = 0; y < rayt.getCanvas()->getH(); ++y) {
            RGB color = rayt.getCanvas()->getPixel( x, y );
            bmp.setPixel( x, y, color.r, color.g, color.b );
        }
    }
    bmp.save( "out.bmp" );
    delete cam, scene;
    return 0;
}