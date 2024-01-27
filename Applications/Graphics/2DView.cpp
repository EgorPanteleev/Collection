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
    std::vector<Sphere*> spheres;
    std::vector<Light*> ligths;
//    spheres.push_back(new Sphere(50, Vector3f(120, 120, 300), RGB(255,0 ,0 )));
//    spheres.push_back(new Sphere(20, Vector3f(40, 40, 400), RGB(0, 255, 0)));
//    spheres.push_back(new Sphere(30, Vector3f(-20, -20, 1000), RGB(0, 0, 255)));
//    spheres.push_back(new Sphere(1500, Vector3f(0, 0, 3000), RGB(255, 255, 0)));

    spheres.push_back(new Sphere(50, Vector3f(120, 120, 300), RGB(255,0 ,0 )));
    spheres.push_back(new Sphere(20, Vector3f(40, 40, 400), RGB(0, 255, 0)));
    spheres.push_back(new Sphere(30, Vector3f(-20, -20, 1000), RGB(0, 0, 255)));
    spheres.push_back(new Sphere(1500, Vector3f(0, 0, 3000), RGB(255, 255, 0)));
    for ( auto s: spheres ) {
        scene->shapes.push_back( s );
    }
    Light* l1 = new Light();
    l1->origin = Vector3f(100,120,190);
    l1->intensity = 0.8;
    ligths.push_back(l1);
    Light* l2 = new Light();
    l2->origin = Vector3f(-300,0,-200);
    l2->intensity = 0.4;
    ligths.push_back(l2);
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