#include <iostream>
#include "RayTracer.h"
#include "Scene.h"
#include "Camera.h"
#include "Sphere.h"
#include "Image.h"
#include <time.h>
#include "Cube.h"
#include "OBJShape.h"
#define GRAY RGB( 210, 210, 210 )
#define RED RGB( 255, 0, 0 )
#define GREEN RGB( 0, 255, 0 )
#define BLUE RGB( 0, 0, 255 )
#define YELLOW RGB( 255, 255, 0 )
#define BROWN RGB( 150, 75, 0 )
#define PINK RGB( 255,105,180 )
#define DARK_BLUE RGB(65,105,225)
int main() {
    //Camera* cam = new Camera( Vector3f(0,0,-160000000), Vector3f(0,0,1), 8000,3200,2000 );
    Camera* cam = new Camera( Vector3f(0, 0,0), Vector3f(0,0,1), 6000,3200,2000 );
    Scene* scene = new Scene();
    RayTracer rayt( cam, scene );
    std::vector<Shape*> shapes;
    std::vector<Material> materials;
    std::vector<Light*> ligths;
//    shapes.push_back(new Sphere(50, Vector3f(120, 120, 300)));
//    materials.emplace_back( RGB( 255, 0, 0), 10 , 0 );
//
//    shapes.push_back(new Sphere(20, Vector3f(-20, 0, 640)));
//    materials.emplace_back( RGB( 0, 255, 0), 10 , 0.7 );
//
//    shapes.push_back(new Sphere(30, Vector3f(-20, -20, 1000)));
//    materials.emplace_back( RGB( 0, 0, 255), 10 , 0 );

//    shapes.push_back(new Sphere(1500, Vector3f(0, 0, 3000)));
//    materials.emplace_back( RGB( 255, 255, 0), 10 , 0 );
//

//    shapes.push_back(new Sphere(5, Vector3f(0, 0, 500)));
//    materials.emplace_back( RGB( 120, 255, 0), 10 , 0 );
//////right
//    shapes.push_back(new Cube( Vector3f(80, -50, 0), Vector3f(100, 50, 600) ) );
//    materials.emplace_back( GREEN, 1 , 0 );
//////left
//    shapes.push_back(new Cube( Vector3f(-100, -50, 0), Vector3f(-80, 50, 600) ) );
//    materials.emplace_back( RED, 1 , 0 );
//////back
//    shapes.push_back(new Cube( Vector3f(-100, -50, 600), Vector3f(100, 50, 600) ) );
//    materials.emplace_back( YELLOW, 1 , 0 );
//////front
//    shapes.push_back(new Cube( Vector3f(-100, -50, 0), Vector3f(100, 50, 0) ) );
//    materials.emplace_back( PINK, 1 , 0 );
//////down
//    shapes.push_back(new Cube( Vector3f(-100, -70, 0), Vector3f(100, -50, 620) ) );
//    materials.emplace_back( DARK_BLUE, 1 , 0 );
//////up
//    shapes.push_back(new Cube( Vector3f(-100, 50, 0), Vector3f(100, 70, 620) ) );
//    materials.emplace_back( BROWN, 1 , 0 );
//// RAND BLOCK
    auto* randBlockForward = new Cube( Vector3f(-15, -50, 310), Vector3f(15, -30, 340) );
    randBlockForward->moveTo( Vector3f(0, -40, 325) );
    randBlockForward->scaleTo( Vector3f(20,90,20) );
    randBlockForward->rotate( Vector3f( 0,1,0), 45);
    shapes.push_back(randBlockForward );
    materials.emplace_back( GRAY, 1 , 0 );

//    auto* randBlockBackward = new Cube( Vector3f(15, -50, -310), Vector3f(-15, -30, -340) );
//    //randBlock->move( Vector3f(-30,0,0 ));
//    //randBlock->rotate( Vector3f( 0,1,0), 45);
//    shapes.push_back(randBlockBackward );
//    materials.emplace_back( GRAY, 1 , 0 );
//
//    auto* randBlockLeft = new Cube( Vector3f(-300, -30, -15), Vector3f(-340, -50, 15) );
//    //randBlock->move( Vector3f(-30,0,0 ));
//    //randBlock->rotate( Vector3f( 0,1,0), 45);
//    shapes.push_back(randBlockLeft );
//    materials.emplace_back( GRAY, 1 , 0 );
//
//    auto* randBlockRight = new Cube( Vector3f(300, -50, -15), Vector3f(340, -30, 15) );
//    //randBlock->move( Vector3f(-30,0,0 ));
//    //randBlock->rotate( Vector3f( 0,1,0), 45);
//    shapes.push_back(randBlockRight );
//    materials.emplace_back( GRAY, 1 , 0 );
//
//    auto* randBlockUp = new Cube( Vector3f(-15, 300, -15), Vector3f(15, 340, 15) );
//    //randBlock->move( Vector3f(-30,0,0 ));
//    //randBlock->rotate( Vector3f( 0,1,0), 45);
//    shapes.push_back(randBlockUp );
//    materials.emplace_back( GRAY, 1 , 0 );
//
//    auto* randBlockDown = new Cube( Vector3f(15, -300, -15), Vector3f(-15, -340, 15) );
//    //randBlock->move( Vector3f(-30,0,0 ));
//    //randBlock->rotate( Vector3f( 0,1,0), 45);
//    shapes.push_back( randBlockDown );
//    materials.emplace_back( GRAY, 1 , 0 );


//    auto* randBlock1 = new Cube( Vector3f(-15, -50, 310), Vector3f(15, -30, 340) );
//    randBlock1->move( Vector3f(30,0,0 ));
//    //randBlock->rotate( Vector3f( 0,1,0), 45);
//    shapes.push_back(randBlock1 );
//    materials.emplace_back( GRAY, 1 , 0 );


//    shapes.push_back(new Cube( Vector3f(-12, -19, 593), Vector3f(10, -17, 600) ) );
//    materials.emplace_back( YELLOW, 1 , 0 );
//
//    shapes.push_back(new Cube( Vector3f(-10, -17, 595), Vector3f(10, 17, 595) ) );
//    materials.emplace_back( YELLOW, 1 , 0 );
//
//    shapes.push_back(new Cube( Vector3f(-10, -17, 595), Vector3f(10, 17, 595) ) );
//    materials.emplace_back( YELLOW, 1 , 0 );
//
//    shapes.push_back(new Cube( Vector3f(-10, -17, 595), Vector3f(10, 17, 595) ) );
//    materials.emplace_back( YELLOW, 1 , 0 );

//    shapes.push_back(new Cube( Vector3f(-13, -30, 595), Vector3f(13, 30, 595) ) );
//    materials.emplace_back( BLUE, 1 , 0.8 );

//    auto* rat = new OBJShape( "C:/Users/igor/CLionProjects/Collection/Modules/model.obj");
//    //rat->rotate( Vector3f( 0,0,1),45);
//    rat->rotate( Vector3f( 1,0,0),270);
//    rat->rotate( Vector3f( 0,1,0),35);
//    rat->move( Vector3f( -16000000,0,0) );
//    shapes.push_back( rat );
//    materials.emplace_back( BROWN, 1 , 0 );
//
//    auto* rat1 = new OBJShape( "C:/Users/igor/CLionProjects/Collection/Modules/model.obj");
//    //rat->rotate( Vector3f( 0,0,1),45);
//    rat1->rotate( Vector3f( 1,0,0),270);
//    rat1->rotate( Vector3f( 0,1,0),155);
//    rat1->move( Vector3f( 15000000,0,0) );
//    shapes.push_back( rat1 );
//    materials.emplace_back( PINK, 1 , 0 );

    for ( int i = 0; i < shapes.size(); ++i ) {
        scene->objects.push_back( new Object( shapes[i], materials[i] ) );
    }
//    ligths.push_back( new Light( Vector3f(0,45,3000000000), 1));
//    ligths.push_back( new Light( Vector3f(0,45,3000000000), 1));
    ligths.push_back( new Light( Vector3f(0,45,300), 0.6));
    ligths.push_back( new Light( Vector3f(0,0,-900), 0.6));
    ligths.push_back( new Light( Vector3f(0,0,900), 0.6));
//    ligths.push_back( new Light( Vector3f(-75,35,595), 0.2));
//    ligths.push_back( new Light( Vector3f(75,35,595), 0.2));
//    ligths.push_back( new Light( Vector3f(-75,35,5), 0.2));
//    ligths.push_back( new Light( Vector3f(75,35,5), 0.2));
//    ligths.push_back( new Light( Vector3f(100,120,20), 0.5));
//    ligths.push_back( new Light( Vector3f(300,0,200), 0.4));
//    ligths.push_back( new Light( Vector3f(300,0,700), 0.4));
    for ( auto l: ligths ) {
        scene->lights.push_back( l );
    }
    clock_t start = clock();
    rayt.traceAllRaysWithThreads( 50 );
    clock_t end = clock();
    double seconds = (double)(end - start) / CLOCKS_PER_SEC;
    printf("The time with multithreading: %f seconds\n", seconds);
    Bitmap bmp(rayt.getCanvas()->getW(), rayt.getCanvas()->getH());
    for (int x = 0; x < rayt.getCanvas()->getW(); ++x) {
        for (int y = 0; y < rayt.getCanvas()->getH(); ++y) {
            RGB color = rayt.getCanvas()->getPixel( x, y );
            bmp.setPixel( x, y, color.r, color.g, color.b );
        }
    }
    bmp.save( "out.bmp" );
//    start = clock();
//    rayt.traceAllRays();
//    end = clock();
//    seconds = (double)(end - start) / CLOCKS_PER_SEC;
//    printf("The time without multithreading: %f seconds\n", seconds);
    delete cam, scene;
    return 0;
}