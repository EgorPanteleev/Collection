#include <iostream>
#include "RayTracer.h"
#include "Scene.h"
#include "Camera.h"
#include "Sphere.h"
#include "Image.h"
#include <ctime>
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
#define CYAN RGB( 0, 255, 255)


//// RAND BLOCK
//    auto* randBlockForward = new Cube( Vector3f(-15, -50, 310), Vector3f(15, -30, 340) );
//    randBlockForward->moveTo( Vector3f(0, -40, 325) );
//    randBlockForward->scaleTo( Vector3f(20,90,20) );
//    randBlockForward->rotate( Vector3f( 0,1,0), 45);
//    shapes.push_back(randBlockForward );
//    materials.emplace_back( GRAY, 1 , 0 );

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


//void initCanvas(Canvas* canvas, int w, int h ) {
//    canvas = new Canvas( w, h );
//}

void loadScene( Scene* scene, std::vector <Shape*>& shapes, std::vector<Light*>& lights, std::vector <Material>& materials ) {
    for ( int i = 0; i < shapes.size(); ++i ) {
        scene->objects.push_back( new Object( shapes[i], materials[i] ) );
    }

    for ( auto l: lights ) {
        scene->lights.push_back( l );
    }
}

void sphereScene( RayTracer*& rayTracer, int w, int h ) {
    Camera* cam = new Camera( Vector3f(0, 0,-10000 ), Vector3f(0,0,1), 6000,3200,2000 );
    Scene* scene = new Scene();
    Canvas* canvas = new Canvas(w, h );

    std::vector<Shape*> shapes;
    std::vector<Material> materials;
    std::vector<Light*> lights;

    shapes.push_back(new Sphere(1500, Vector3f(0, 0, 3000)));
    materials.emplace_back( YELLOW, 0 , 0 );

    shapes.push_back(new Sphere(300, Vector3f(2121, 0, 2250)));
    materials.emplace_back( RED, 0 , 0 );

    shapes.push_back(new Sphere(300, Vector3f(1030, 0, 1000)));
    materials.emplace_back( GREEN, 0 , 0 );

    shapes.push_back(new Sphere(300, Vector3f(-2121, 0, 2250)));
    materials.emplace_back( PINK, 0 , 0 );

    shapes.push_back(new Sphere(300, Vector3f(-1030, 0, 1000)));
    materials.emplace_back( CYAN, 0 , 0 );


    //lights.push_back( new Light( Vector3f(-3500,0,0 ), 0.004 ));
    lights.push_back( new Light( Vector3f(-1000,0,0 ), 0.004 ));
    loadScene( scene, shapes, lights, materials );
    rayTracer = new RayTracer( cam, scene, canvas );
}


void roomScene( RayTracer*& rayTracer, int w, int h ) {
    Camera* cam = new Camera( Vector3f(0,0,0 ), Vector3f(0,0,1), 6000,3200,2000 );
    Scene* scene = new Scene();
    Canvas* canvas = new Canvas( w, h );

    std::vector<Shape*> shapes;
    std::vector<Material> materials;
    std::vector<Light*> lights;
////right
    shapes.push_back(new Cube( Vector3f(80, -50, 0), Vector3f(100, 50, 600) ) );
    materials.emplace_back( GRAY, 1 , 0 );
////left
    shapes.push_back(new Cube( Vector3f(-100, -50, 0), Vector3f(-80, 50, 600) ) );
    materials.emplace_back( GRAY, 1 , 0 );
////back
    shapes.push_back(new Cube( Vector3f(-100, -50, 600), Vector3f(100, 50, 600) ) );
    materials.emplace_back( GRAY, 1 , 0 );
////front
    shapes.push_back(new Cube( Vector3f(-100, -50, 0), Vector3f(100, 50, 0) ) );
    materials.emplace_back( GRAY, 1 , 0 );
////down
    shapes.push_back(new Cube( Vector3f(-100, -70, 0), Vector3f(100, -50, 620) ) );
    materials.emplace_back( GRAY, 1 , 0 );
////up
    shapes.push_back(new Cube( Vector3f(-100, 50, 0), Vector3f(100, 70, 620) ) );
    materials.emplace_back( GRAY, 1 , 0 );

////RAND BLOCK
    auto* randBlockForward = new Cube( Vector3f(-15, -50, 310), Vector3f(15, -30, 340) );
    randBlockForward->moveTo( Vector3f(0, -40, 325) );
    randBlockForward->scaleTo( Vector3f(20,90,20) );
    randBlockForward->rotate( Vector3f( 0,1,0), 45);
    randBlockForward->move( Vector3f(-10,0,0));
    shapes.push_back(randBlockForward );
    materials.emplace_back( RED, 1 , 0 );
////LIGHTS
    lights.push_back( new Light( Vector3f(-75,35,595), 0.15));
    lights.push_back( new Light( Vector3f(75,35,595), 0.15));
    lights.push_back( new Light( Vector3f(-75,35,5), 0.15));
    lights.push_back( new Light( Vector3f(75,35,5), 0.15));
////LOADING...
    loadScene( scene, shapes, lights, materials );
    rayTracer = new RayTracer( cam, scene, canvas );
}


void ratScene( RayTracer*& rayTracer, int w, int h ) {
    Camera* cam = new Camera( Vector3f(0,0,0 ), Vector3f(0,0,1), 600,3200,2000 );
    Scene* scene = new Scene();
    Canvas* canvas = new Canvas( w, h );
    std::vector<Shape*> shapes;
    std::vector<Material> materials;
    std::vector<Light*> lights;
    //TODO normalize coords to see it on camera

    auto* rat = new OBJShape( "/home/auser/dev/src/Collection/Models/model.obj");
    //rat->rotate( Vector3f( 0, 0, 1), 45 );
    rat->rotate( Vector3f( 1,0,0),270);
    rat->rotate( Vector3f( 0,1,0),35);
    rat->move( Vector3f( 0,0,15000000) );
    shapes.push_back( rat );
    materials.emplace_back( RGB( 130, 130, 130 ), 1 , 0 );

//    auto* rat1 = new OBJShape( "C:/Users/igor/CLionProjects/Collection/Modules/model.obj");
//    //rat->rotate( Vector3f( 0,0,1),45);
//    rat1->rotate( Vector3f( 1,0,0),270);
//    rat1->rotate( Vector3f( 0,1,0),155);
//    rat1->move( Vector3f( 15000000,0,0) );
//    shapes.push_back( rat1 );
//    materials.emplace_back( PINK, 1 , 0 );

    lights.push_back( new Light( Vector3f(2000000 ,0,0), 1));
    loadScene( scene, shapes, lights, materials );
    rayTracer = new RayTracer( cam, scene, canvas );
}

void saveToBMP( Canvas* canvas, std::string fileName ) {
    Bitmap bmp(canvas->getW(), canvas->getH());
    for (int x = 0; x < canvas->getW(); ++x) {
        for (int y = 0; y < canvas->getH(); ++y) {
            RGB color = canvas->getPixel( x, y );
            bmp.setPixel( x, y, color.r, color.g, color.b );
        }
    }
    bmp.save( fileName );
}

int main() {
    RayTracer* rayTracer = nullptr;
    //int w = 240 ; int h = 150;
    int w = 960 ; int h = 600;
    //int w = 1920 ; int h = 1200;
    //int w = 3200 ; int h = 2000;
// room scene ( 960x600 ) - 18.1 / 15.5 / 9.7 / 9.3 / 7.3
// room scene ( 3200x2000 ) - idk / 95 /
// rat scene ( 3200x2000 ) - 100 / 79 /
    //sphereScene( rayTracer, w, h );
    roomScene( rayTracer, w, h );
    //ratScene( rayTracer, w, h );
    clock_t start = clock();
    //rayt.traceAllRaysWithThreads( 1);
    rayTracer->traceAllRays();
    clock_t end = clock();
    double seconds = (double)(end - start) / CLOCKS_PER_SEC;
    printf("The time with multithreading: %f seconds\n", seconds);
    saveToBMP( rayTracer->getCanvas(), "out.bmp" );
   //delete
   //TODO mb need init rayTracer more
   //TODO think about camera, i think its bad right now
    return 0;
}