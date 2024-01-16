#include <iostream>
#include "RayTracer.h"
#include "Scene.h"
#include "Camera.h"
#include "Sphere.h"
#include "Image.h"

int main() {
    Camera* cam = new Camera( Vector3f(0,0,-1), Vector3f(0,0,1), 1000,1000,1000 );
    Scene* scene = new Scene();
    RayTracer rayt( cam, scene );
    std::vector<Sphere*> spheres;
    std::vector<Light*> ligths;
    spheres.push_back(new Sphere(50, Vector3f(120, 120, 300), RGB(255,0 ,0 )));
    spheres.push_back(new Sphere(20, Vector3f(40, 40, 400), RGB(0, 255, 0)));
    spheres.push_back(new Sphere(30, Vector3f(-20, -20, 500), RGB(0, 0, 255)));
    for ( auto s: spheres ) {
        scene->shapes.push_back( s );
    }
    Light* l1 = new Light();
    l1->origin = Vector3f(-150,300,100);
    l1->intensity = 0.8;
    ligths.push_back(l1);
    Light* l2 = new Light();
    l2->origin = Vector3f(300,0,100);
    l2->intensity = 0.6;
    //ligths.push_back(l2);
    for ( auto l: ligths ) {
        scene->lights.push_back( l );
    }
    rayt.traceAllRays();
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