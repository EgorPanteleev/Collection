#include <iostream>
#include "RayTracer.h"
#include "Scene.h"
#include "Camera.h"
#include "SphereMesh.h"
#include "Image.h"
#include <ctime>
#include "CubeMesh.h"
#include "BaseMesh.h"
#include "TriangularMesh.h"
#include "Material.h"
#include "PointLight.h"
#include "SpotLight.h"
#include "cstdlib"
#include "Denoiser.h"
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

void loadScene( Scene* scene, std::vector <BaseMesh*>& meshes, std::vector<Light*>& lights ) {
    for ( const auto& mesh: meshes ) {
        scene->meshes.push_back( mesh );
    }
    for ( const auto& light: lights ) {
        scene->lights.push_back( light );
    }
}

void sphereScene( RayTracer*& rayTracer, int w, int h, int d, int numAS, int numLS ) {
    Camera* cam = new Camera( Vector3f(0, 0,-10000 ), Vector3f(0,0,1), 6000,3200,2000 );
    Scene* scene = new Scene();
    Canvas* canvas = new Canvas(w, h );

    std::vector<BaseMesh*> meshes;
    std::vector<Light*> lights;

    meshes.push_back(new SphereMesh(1500, Vector3f(0, 0, 3000), {YELLOW, 0 , 0 }));

    meshes.push_back(new SphereMesh(300, Vector3f(2121, 0, 2250), {RED, 0 , 0}));

    meshes.push_back(new SphereMesh(300, Vector3f(1030, 0, 1000),{GREEN, 0 , 0}));

    meshes.push_back(new SphereMesh(300, Vector3f(-2121, 0, 2250),{PINK, 0 , 0}));

    meshes.push_back(new SphereMesh(300, Vector3f(-1030, 0, 1000),{CYAN, 0 , 0}));



    //lights.push_back( new PointLight( Vector3f(-3500,0,0 ), 0.004 ));
    lights.push_back( new PointLight( Vector3f(-1000,0,0 ), 0.004 ));
    loadScene( scene, meshes, lights );
    rayTracer = new RayTracer( cam, scene, canvas, d, numAS, numLS );
}

void netRoomScene( RayTracer*& rayTracer, int w, int h, int d, int numAS, int numLS ) {
    Camera* cam = new Camera( Vector3f(0,10,0 ), Vector3f(0,0,1), 2400,3200,2000 );
    //Camera* cam = new Camera( Vector3f(0,0,0 ), Vector3f(0,0,1), 6000,3200,2000 );
    Scene* scene = new Scene();
    Canvas* canvas = new Canvas( w, h );

    std::vector<BaseMesh*> meshes;
    std::vector<Light*> lights;
    float roomRefl = 0;
////right
    meshes.push_back( new CubeMesh( Vector3f(70, -50, 0), Vector3f(80, 70, 600),
                                    { GREEN, -1 , roomRefl } ) );
////left
    meshes.push_back(new CubeMesh( Vector3f(-80, -50, 0), Vector3f(-70, 70, 600),
                                   { RED, -1 , roomRefl } ) );
////front
    meshes.push_back(new CubeMesh( Vector3f(-100, -50, 290), Vector3f(100, 70, 300),
                                   { GRAY, -1 , roomRefl } ) );
////back
    meshes.push_back(new CubeMesh( Vector3f(-100, -50, -10), Vector3f(100, 70, 0),
                                   { GRAY, -1 , roomRefl } ) );
////down
    meshes.push_back(new CubeMesh( Vector3f(-100, -70, 0), Vector3f(100, -50, 620),
                                   { GRAY, -1 , roomRefl } ) );
////up
    meshes.push_back(new CubeMesh( Vector3f(-100, 70, roomRefl), Vector3f(100, 90, 620),
                                   { GRAY, -1 , 0 } ) );

////RAND BLOCK
    auto* randBlockForward = new CubeMesh( Vector3f(-15, -50, 310), Vector3f(15, -30, 340) );
    randBlockForward->moveTo( Vector3f(0, -40, 325) );
    randBlockForward->scaleTo( Vector3f(30,100,30) );
    randBlockForward->rotate( Vector3f( 0,1,0), 25);
    randBlockForward->move( Vector3f(30,0,0));
    randBlockForward->setMaterial({GRAY, -1 , 0});
    randBlockForward->move( Vector3f(-10,0,-150));
    //randBlockForward->scaleTo( 200 );
    meshes.push_back(randBlockForward );

    auto* randBlockForward2 = new CubeMesh( Vector3f(-15, -50, 310), Vector3f(15, -30, 340) );
    randBlockForward2->moveTo( Vector3f(0, -40, 325) );
    randBlockForward2->scaleTo( Vector3f(30,260,30) );
    randBlockForward2->rotate( Vector3f( 0,1,0), -25);
    randBlockForward2->move( Vector3f(35,0,0));
    randBlockForward2->setMaterial({GRAY, -1 , 0.8});
    randBlockForward2->move( Vector3f(-50,0,-100));
    //randBlockForward2->scaleTo( 200 );
    meshes.push_back(randBlockForward2 );

////LIGHTS

    //lights.push_back( new PointLight( Vector3f(0,65,150), 0.55));
    int lightWidth = 20;
    lights.push_back( new SpotLight( Vector3f(0 - lightWidth,65,180 - lightWidth), Vector3f(0 + lightWidth,65,180 + lightWidth), 0.7));

////LOADING...
    loadScene( scene, meshes, lights );
    rayTracer = new RayTracer( cam, scene, canvas, d, numAS, numLS );
}

void simpleRoomScene( RayTracer*& rayTracer, int w, int h, int d, int numAS, int numLS ) {
    Camera* cam = new Camera( Vector3f(0,0,300 ), Vector3f(0,0,1), 2400,3200,2000 );
    //Camera* cam = new Camera( Vector3f(0,0,0 ), Vector3f(0,0,1), 6000,3200,2000 );
    Scene* scene = new Scene();
    Canvas* canvas = new Canvas( w, h );

    std::vector<BaseMesh*> meshes;
    std::vector<Light*> lights;
////right
    meshes.push_back( new CubeMesh( Vector3f(80, -50, 0), Vector3f(100, 50, 600),
                                    { GRAY, -1 , 0 } ) );
////left
    meshes.push_back(new CubeMesh( Vector3f(-100, -50, 0), Vector3f(-80, 50, 600),
                                   { GRAY, -1 , 0 } ) );
////front
    meshes.push_back(new CubeMesh( Vector3f(-100, -50, 600), Vector3f(100, 50, 610),
                                   { GRAY, -1 , 0 } ) );
////back
    meshes.push_back(new CubeMesh( Vector3f(-100, -50, -10), Vector3f(100, 50, 0),
                                   { GRAY, -1 , 0 } ) );
////down
    meshes.push_back(new CubeMesh( Vector3f(-100, -70, 0), Vector3f(100, -50, 620),
                                   { GRAY, -1 , 0 } ) );
////up
    meshes.push_back(new CubeMesh( Vector3f(-100, 50, 0), Vector3f(100, 70, 620),
                                   { GRAY, -1 , 0 } ) );

////RAND BLOCK
    auto* randBlockForward = new CubeMesh( Vector3f(-15, -50, 310), Vector3f(15, -30, 340) );
    randBlockForward->moveTo( Vector3f(0, -40, 325) );
    randBlockForward->scaleTo( Vector3f(20,90,20) );
    randBlockForward->rotate( Vector3f( 0,1,0), 45);
    randBlockForward->move( Vector3f(30,0,0));
    randBlockForward->setMaterial({RED, 1 , 0});
    randBlockForward->move( Vector3f(-50,0,180));
    //randBlockForward->scaleTo( 200 );
    meshes.push_back(randBlockForward );

////LIGHTS

    lights.push_back( new PointLight( Vector3f(0,25,450), 0.45));

//    lights.push_back( new PointLight( Vector3f(-75,35,595), 0.15));
//    lights.push_back( new PointLight( Vector3f(75,35,595), 0.15));
//    lights.push_back( new PointLight( Vector3f(-75,35,5), 0.15));
//    lights.push_back( new PointLight( Vector3f(75,35,5), 0.15));
////LOADING...
    loadScene( scene, meshes, lights );
    rayTracer = new RayTracer( cam, scene, canvas, d, numAS, numLS );
}


void roomScene( RayTracer*& rayTracer, int w, int h, int d, int numAS, int numLS ) {
    Camera* cam = new Camera( Vector3f(0,0,300 ), Vector3f(0,0,1), 3000,3200,2000 );
    //Camera* cam = new Camera( Vector3f(0,0,0 ), Vector3f(0,0,1), 6000,3200,2000 );
    Scene* scene = new Scene();
    Canvas* canvas = new Canvas( w, h );

    std::vector<BaseMesh*> meshes;
    std::vector<Light*> lights;
////right
    meshes.push_back( new CubeMesh( Vector3f(80, -50, 0), Vector3f(100, 50, 600),
                                    { GRAY, -1 , 0 } ) );
////left
    meshes.push_back(new CubeMesh( Vector3f(-100, -50, 0), Vector3f(-80, 50, 600),
                                   { GRAY, -1 , 0 } ) );
////front
    meshes.push_back(new CubeMesh( Vector3f(-100, -50, 600), Vector3f(100, 50, 610),
                                   { GRAY, -1 , 0 } ) );
////back
    meshes.push_back(new CubeMesh( Vector3f(-100, -50, -10), Vector3f(100, 50, 0),
                                   { GRAY, -1 , 0 } ) );
////down
    meshes.push_back(new CubeMesh( Vector3f(-100, -70, 0), Vector3f(100, -50, 620),
                                   { GRAY, -1 , 0 } ) );
////up
    meshes.push_back(new CubeMesh( Vector3f(-100, 50, 0), Vector3f(100, 70, 620),
                                   { GRAY, -1 , 0 } ) );

////RAND BLOCK
    auto* randBlockForward = new CubeMesh( Vector3f(-15, -50, 310), Vector3f(15, -30, 340) );
    randBlockForward->moveTo( Vector3f(0, -40, 325) );
    randBlockForward->scaleTo( Vector3f(20,90,20) );
    randBlockForward->rotate( Vector3f( 0,1,0), 45);
    randBlockForward->move( Vector3f(30,0,0));
    randBlockForward->setMaterial({RED, 1 , 0});
    randBlockForward->move( Vector3f(-95,0,200));
    //randBlockForward->scaleTo( 200 );
    meshes.push_back(randBlockForward );

////DOG
    auto* dog = new TriangularMesh();
    dog->loadMesh( "/home/auser/dev/src/Collection/Models/dog/model.obj" );
    //dog->rotate( Vector3f( 0, 0, 1), 45 );
    dog->rotate( Vector3f( 1,0,0),-90);
    dog->rotate( Vector3f( 0,1,0),-235);
    dog->scaleTo( 50 );
    dog->move( Vector3f( -10,0,520) );
    dog->setMinPoint( { 0, -50, 0 }, 1 );
    dog->setMaterial( { GRAY, 1 , 0 } );
    meshes.push_back( dog );
    ////SKS
    auto* sks = new TriangularMesh();
    sks->loadMesh( "/home/auser/dev/src/Collection/Models/sks/model.obj" );
    sks->rotate( Vector3f( 0, 0, 1), 60 );
    //sks->rotate( Vector3f( 1,0,0),100);
    sks->rotate( Vector3f( 0,1,0), -120);
    sks->scaleTo(45);
    sks->setMinPoint( { 0, -50, 0 }, 1 );
    sks->move( Vector3f( 39,0,305) );
    sks->move( Vector3f(-95,0,200));
    sks->setMaterial( { GRAY, 1 , 0 } );
    meshes.push_back( sks );

    ////TABLE
    auto* table = new TriangularMesh();
    table->loadMesh( "/home/auser/dev/src/Collection/Models/table/model.obj" );
    //table->rotate( Vector3f( 0, 0, 1), 45 );
    //table->rotate( Vector3f( 1,0,0),270);
    table->rotate( Vector3f( 0,1,0),0);
    table->scaleTo( 75 );
    table->move( Vector3f( 40,40,500) );
    table->setMinPoint({ 0, -50, 0 }, 1);
    table->setMaxPoint({ 0, 0, 600 }, 2);
    table->setMaxPoint({ 80, 0, 0 }, 0);
    table->setMaterial( { GRAY, 1 , 0 } );
    meshes.push_back( table );

    // -80, 80 ; -50, 50; 0, 600
    ////Mirror
    Vector3f moveVec = { 35,-5,0};
    RGB colorRam = GRAY;
    auto* mirrorBottom = new CubeMesh( Vector3f(0, 0, 595), Vector3f(30, 2, 600) );
    mirrorBottom->setMaterial( { colorRam, 1 , 0 } );
    mirrorBottom->move(moveVec);
    meshes.push_back( mirrorBottom );

    auto* mirrorLeft = new CubeMesh( Vector3f(0, 2, 595), Vector3f(2, 37, 600) );
    mirrorLeft->setMaterial( { colorRam, 1 , 0 } );
    mirrorLeft->move(moveVec);
    meshes.push_back( mirrorLeft );

    auto* mirrorRight = new CubeMesh( Vector3f(28, 2, 595), Vector3f(30, 37, 600) );
    mirrorRight->setMaterial( { colorRam, 1 , 0 } );
    mirrorRight->move(moveVec);
    meshes.push_back( mirrorRight );

    auto* mirrorTop = new CubeMesh( Vector3f(2, 35, 595), Vector3f(28, 37, 600) );
    mirrorTop->setMaterial( { colorRam, 1 , 0 } );
    mirrorTop->move(moveVec);
    meshes.push_back( mirrorTop );

    auto* mirror = new CubeMesh( Vector3f(2, 2, 598), Vector3f(30, 35, 600) );
    mirror->setMaterial( { GRAY, 1 , 1 } );
    mirror->move(moveVec);
    meshes.push_back( mirror );

    //CUBE
    auto* cube = new CubeMesh( Vector3f(2, 2, 592), Vector3f(12, 4, 602) );
    cube->setMaterial( { GRAY, 1 , 0 } );
    cube->move({14,-14,-20});
    meshes.push_back( cube );

    auto* plane = new TriangularMesh();
    plane->loadMesh( "/home/auser/dev/src/Collection/Models/plane/model.obj" );
    //plane->rotate( Vector3f( 0, 0, 1), 10 );
    plane->rotate( Vector3f( 1,0,0),-15);
    plane->rotate( Vector3f( 0,1,0),-140);
    plane->move( Vector3f( 20,0,576) );
    plane->setMinPoint( Vector3f( 0,-63,0), 1 );
    plane->scaleTo( 15 );
    plane->setMaterial( { GRAY, 1 , 0 } );
    meshes.push_back( plane );

////LIGHTS

    //lights.push_back( new PointLight( Vector3f(0,25,450), 0.55));
    int lightWidth = 20;
    lights.push_back( new SpotLight( Vector3f(0 - lightWidth,45,480 - lightWidth), Vector3f(0 + lightWidth,45,480 + lightWidth), 0.70));

//    lights.push_back( new PointLight( Vector3f(-75,35,595), 0.15));
//    lights.push_back( new PointLight( Vector3f(75,35,595), 0.15));
//    lights.push_back( new PointLight( Vector3f(-75,35,5), 0.15));
//    lights.push_back( new PointLight( Vector3f(75,35,5), 0.15));
////LOADING...
    loadScene( scene, meshes, lights );
    rayTracer = new RayTracer( cam, scene, canvas, d, numAS, numLS );
}


void ratScene( RayTracer*& rayTracer, int w, int h, int d, int numAS, int numLS ) {
    Camera* cam = new Camera( Vector3f(0,0,0 ), Vector3f(0,0,1), 6000,3200,2000 );
    Scene* scene = new Scene();
    Canvas* canvas = new Canvas( w, h );
    std::vector<BaseMesh*> meshes;
    std::vector<Light*> lights;

    auto* rat = new TriangularMesh();
    rat->loadMesh( "/home/auser/dev/src/Collection/Models/rat/model.obj" );
    //rat->rotate( Vector3f( 0, 0, 1), 45 );
    rat->rotate( Vector3f( 1,0,0),270);
    rat->rotate( Vector3f( 0,1,0),145);
    rat->move( Vector3f( 0,0,500) );
    rat->setMaterial( { RGB( 130, 130, 130 ), 1 , 0 } );
    meshes.push_back( rat );

    lights.push_back( new PointLight( Vector3f(20 ,0,0), 0.5));
    loadScene( scene, meshes, lights );
    rayTracer = new RayTracer( cam, scene, canvas, d, numAS, numLS );
}


void tableScene( RayTracer*& rayTracer, int w, int h, int d, int numAS, int numLS ) {
    Camera* cam = new Camera( Vector3f(0,0,0 ), Vector3f(0,0,1), 6000,3200,2000 );
    Scene* scene = new Scene();
    Canvas* canvas = new Canvas( w, h );
    std::vector<BaseMesh*> meshes;
    std::vector<Light*> lights;

    auto* table = new TriangularMesh();
    table->loadMesh( "/home/auser/dev/src/Collection/Models/table/model.obj" );
    //table->rotate( Vector3f( 0, 0, 1), 45 );
    //table->rotate( Vector3f( 1,0,0),270);
    table->rotate( Vector3f( 0,1,0),-120);
    table->move( Vector3f( 40,40,1000) );
    table->setMaterial( { RGB( 130, 130, 130 ), 1 , 0 } );
    meshes.push_back( table );

    lights.push_back( new PointLight( Vector3f(20 ,0,0), 0.5));
    loadScene( scene, meshes, lights );
    rayTracer = new RayTracer( cam, scene, canvas, d, numAS, numLS );
}

void bookScene( RayTracer*& rayTracer, int w, int h, int d, int numAS, int numLS ) {
    Camera* cam = new Camera( Vector3f(0,0,0 ), Vector3f(0,0,1), 6000,3200,2000 );
    Scene* scene = new Scene();
    Canvas* canvas = new Canvas( w, h );
    std::vector<BaseMesh*> meshes;
    std::vector<Light*> lights;

    auto* book = new TriangularMesh();
    book->loadMesh( "/home/auser/dev/src/Collection/Models/book/model.obj" );
    //book->rotate( Vector3f( 0, 0, 1), 45 );
    book->rotate( Vector3f( 1,0,0),-30);
    book->rotate( Vector3f( 0,1,0),-130);
    book->move( Vector3f( 40,0,1000) );
    book->setMaterial( { RGB( 130, 130, 130 ), 1 , 0 } );
    meshes.push_back( book );

    lights.push_back( new PointLight( Vector3f(20 ,0,0), 0.5));
    loadScene( scene, meshes, lights );
    rayTracer = new RayTracer( cam, scene, canvas, d, numAS, numLS );
}

void sandwichScene( RayTracer*& rayTracer, int w, int h, int d, int numAS, int numLS ) {
    Camera* cam = new Camera( Vector3f(0,0,0 ), Vector3f(0,0,1), 6000,3200,2000 );
    Scene* scene = new Scene();
    Canvas* canvas = new Canvas( w, h );
    std::vector<BaseMesh*> meshes;
    std::vector<Light*> lights;

    auto* sandwich = new TriangularMesh();
    sandwich->loadMesh( "/home/auser/dev/src/Collection/Models/sandwich/model.obj" );
    sandwich->rotate( Vector3f( 0, 0, 1), 180 );
    sandwich->rotate( Vector3f( 1,0,0),-90);
    sandwich->rotate( Vector3f( 0,1,0),-70);
    sandwich->move( Vector3f( 40,0,1000) );
    sandwich->setMaterial( { RGB( 130, 130, 130 ), 1 , 0 } );
    meshes.push_back( sandwich );

    lights.push_back( new PointLight( Vector3f(20 ,0,0), 0.5));
    loadScene( scene, meshes, lights );
    rayTracer = new RayTracer( cam, scene, canvas, d, numAS, numLS );
}

void cartScene( RayTracer*& rayTracer, int w, int h, int d, int numAS, int numLS ) {
    Camera* cam = new Camera( Vector3f(0,0,0 ), Vector3f(0,0,1), 6000,3200,2000 );
    Scene* scene = new Scene();
    Canvas* canvas = new Canvas( w, h );
    std::vector<BaseMesh*> meshes;
    std::vector<Light*> lights;

    auto* cart = new TriangularMesh();
    cart->loadMesh( "/home/auser/dev/src/Collection/Models/telega/model.obj" );
    //cart->rotate( Vector3f( 0, 0, 1), 45 );
    cart->rotate( Vector3f( 1,0,0),-90);
    cart->rotate( Vector3f( 0,1,0),60);
    cart->move( Vector3f( 40,0,1000) );
    cart->setMaterial( { RGB( 130, 130, 130 ), 1 , 0 } );
    meshes.push_back( cart );

    lights.push_back( new PointLight( Vector3f(20 ,0,0), 0.5));
    loadScene( scene, meshes, lights );
    rayTracer = new RayTracer( cam, scene, canvas, d, numAS, numLS );
}

void sksScene( RayTracer*& rayTracer, int w, int h, int d, int numAS, int numLS ) {
    Camera* cam = new Camera( Vector3f(0,0,0 ), Vector3f(0,0,1), 6000,3200,2000 );
    Scene* scene = new Scene();
    Canvas* canvas = new Canvas( w, h );
    std::vector<BaseMesh*> meshes;
    std::vector<Light*> lights;

    auto* sks = new TriangularMesh();
    sks->loadMesh( "/home/auser/dev/src/Collection/Models/sks/model.obj" );
    //sks->rotate( Vector3f( 0, 0, 1), 45 );
    sks->rotate( Vector3f( 1,0,0),-90);
    sks->rotate( Vector3f( 0,1,0),-20);
    sks->move( Vector3f( 0,0,600) );
    sks->setMaterial( { RGB( 130, 130, 130 ), 1 , 0 } );
    meshes.push_back( sks );

    lights.push_back( new PointLight( Vector3f(20 ,0,0), 0.5));
    loadScene( scene, meshes, lights );
    rayTracer = new RayTracer( cam, scene, canvas, d, numAS, numLS );
}

void dogScene( RayTracer*& rayTracer, int w, int h, int d, int numAS, int numLS ) {
    Camera* cam = new Camera( Vector3f(0,0,0 ), Vector3f(0,0,1), 6000,3200,2000 );
    Scene* scene = new Scene();
    Canvas* canvas = new Canvas( w, h );
    std::vector<BaseMesh*> meshes;
    std::vector<Light*> lights;

    auto* dog = new TriangularMesh();
    dog->loadMesh( "/home/auser/dev/src/Collection/Models/dog/model.obj" );
    //dog->rotate( Vector3f( 0, 0, 1), 45 );
    dog->rotate( Vector3f( 1,0,0),-90);
    dog->rotate( Vector3f( 0,1,0),-235);
    dog->move( Vector3f( 0,0,700) );
    dog->setMaterial( { RGB( 130, 130, 130 ), 1 , 0 } );
    meshes.push_back( dog );

    lights.push_back( new PointLight( Vector3f(20 ,0,0), 0.5));
    loadScene( scene, meshes, lights );
    rayTracer = new RayTracer( cam, scene, canvas, d, numAS, numLS );
}

void planeScene( RayTracer*& rayTracer, int w, int h, int d, int numAS, int numLS ) {
    Camera* cam = new Camera( Vector3f(0,0,0 ), Vector3f(0,0,1), 6000,3200,2000 );
    Scene* scene = new Scene();
    Canvas* canvas = new Canvas( w, h );
    std::vector<BaseMesh*> meshes;
    std::vector<Light*> lights;

    auto* plane = new TriangularMesh();
    plane->loadMesh( "/home/auser/dev/src/Collection/Models/plane/model.obj" );
    //dog->rotate( Vector3f( 0, 0, 1), 45 );
    plane->rotate( Vector3f( 1,0,0),10);
    plane->rotate( Vector3f( 0,1,0),-200);
    plane->move( Vector3f( 0,0,600) );
    plane->setMaterial( { RGB( 130, 130, 130 ), 1 , 0 } );
    meshes.push_back( plane );
    lights.push_back( new PointLight( Vector3f(20 ,0,0), 0.5));
    loadScene( scene, meshes, lights );
    rayTracer = new RayTracer( cam, scene, canvas, d, numAS, numLS );
}

void modelRoomScene( RayTracer*& rayTracer, int w, int h, int d, int numAS, int numLS ) {
    Camera* cam = new Camera( Vector3f(0,0,0 ), Vector3f(0,0,1), 6000,3200,2000 );
    Scene* scene = new Scene();
    Canvas* canvas = new Canvas( w, h );
    std::vector<BaseMesh*> meshes;
    std::vector<Light*> lights;

    auto* room = new TriangularMesh();
    room->loadMesh( "/home/auser/dev/src/Collection/Models/cottage/Cottage_FREE.obj" );
    //room->rotate( Vector3f( 0, 0, 1), 45 );
    //room->rotate( Vector3f( 1,0,0),10);
    //room->rotate( Vector3f( 0,1,0),-200);
    room->move( Vector3f( 0,0,0) );
    room->setMaterial( { RGB( 130, 130, 130 ), 1 , 0 } );
    meshes.push_back( room );
    lights.push_back( new PointLight( Vector3f(20 ,0,0), 0.5));
    loadScene( scene, meshes, lights );
    rayTracer = new RayTracer( cam, scene, canvas, d, numAS, numLS );
}
void cottageScene( RayTracer*& rayTracer, int w, int h, int d, int numAS, int numLS ) {
    Camera* cam = new Camera( Vector3f(0,0,0 ), Vector3f(0,0,1), 6000,3200,2000 );
    Scene* scene = new Scene();
    Canvas* canvas = new Canvas( w, h );
    std::vector<BaseMesh*> meshes;
    std::vector<Light*> lights;

    auto* cottage = new TriangularMesh();
    cottage->loadMesh( "/home/auser/dev/src/Collection/Models/underground/underground.obj" );
    //cottage->rotate( Vector3f( 0, 0, 1), 45 );
    //cottage->rotate( Vector3f( 1,0,0),10);
    cottage->rotate( Vector3f( 0,1,0),-260);
    cottage->move( Vector3f( 0,0,800) );
    cottage->setMaterial( { RGB( 130, 130, 130 ), 1 , 0 } );
    meshes.push_back( cottage );
    lights.push_back( new PointLight( Vector3f(20 ,0,0), 0.5));
    loadScene( scene, meshes, lights );
    rayTracer = new RayTracer( cam, scene, canvas, d, numAS, numLS );
}

void carScene( RayTracer*& rayTracer, int w, int h, int d, int numAS, int numLS ) {
    Camera* cam = new Camera( Vector3f(0,0,0 ), Vector3f(0,0,1), 6000,3200,2000 );
    Scene* scene = new Scene();
    Canvas* canvas = new Canvas( w, h );
    std::vector<BaseMesh*> meshes;
    std::vector<Light*> lights;

    auto* car = new TriangularMesh();
    car->loadMesh( "/home/auser/dev/src/Collection/Models/koenigsegg/model.obj" );
    //car->rotate( Vector3f( 0, 0, 1), 45 );
    //car->rotate( Vector3f( 1,0,0),10);
    car->rotate( Vector3f( 0,1,0),-210);
    car->move( Vector3f( 0,0,600) );
    car->setMaterial( { RGB( 130, 130, 130 ), 1 , 0 } );
    meshes.push_back( car );
    lights.push_back( new PointLight( Vector3f(20 ,0,0), 0.5));
    loadScene( scene, meshes, lights );
    rayTracer = new RayTracer( cam, scene, canvas, d, numAS, numLS );
}

void hardScene( RayTracer*& rayTracer, int w, int h, int d, int numAS, int numLS ) {
    Camera* cam = new Camera( Vector3f(0,0,0 ), Vector3f(0,0,1), 6000,3200,2000 );
    Scene* scene = new Scene();
    Canvas* canvas = new Canvas( w, h );
    std::vector<BaseMesh*> meshes;
    std::vector<Light*> lights;
    auto* table = new TriangularMesh();
    table->loadMesh( "/home/auser/dev/src/Collection/Models/table/model.obj" );
    //table->rotate( Vector3f( 0, 0, 1), 20 );
    //table->rotate( Vector3f( 1,0,0),20);
    table->rotate( Vector3f( 0,1,0),-120);
    table->move( Vector3f( 40,40,1200) );
    table->setMaterial( { RGB( 130, 130, 130 ), 1 , 0 } );



    auto* sks = new TriangularMesh();
    sks->loadMesh( "/home/auser/dev/src/Collection/Models/sks/model.obj" );
    sks->rotate( Vector3f( 0, 0, 1), 0 );
    sks->rotate( Vector3f( 1,0,0),0);
    sks->rotate( Vector3f( 0,1,0),0);
    sks->move( Vector3f( -25,19,1080) );
    sks->setMaterial( { RGB( 130, 130, 130 ), 1 , 0 } );

//    meshes.push_back( table );
//    meshes.push_back( sks );

    auto tr1 = table->getTriangles();
    auto tr2 = sks->getTriangles();
    for ( const auto& a: tr1 ) {
        tr2.push_back( a );
    }
    auto asd = new TriangularMesh();
    asd->setTriangles( tr2 );
    asd->setMaterial( { RGB( 130, 130, 130 ), 1 , 0 } );
    asd->rotate( Vector3f( 0, 0, 1), 0 );
    asd->rotate( Vector3f( 1,0,0),-40);
    asd->rotate( Vector3f( 0,1,0),0);
    meshes.push_back( asd );

    lights.push_back( new PointLight( Vector3f(20 ,0,0), 0.5));
    loadScene( scene, meshes, lights );
    rayTracer = new RayTracer( cam, scene, canvas, d, numAS, numLS );
}


void saveToBMP( Canvas* canvas, const std::string& fileName ) {
    Bitmap bmp(canvas->getW(), canvas->getH());
    for (int x = 0; x < canvas->getW(); ++x) {
        for (int y = 0; y < canvas->getH(); ++y) {
            RGB color = canvas->getPixel(x, y);
            //std::cout << color.r << " " << color.g << " " << color.b << std::endl;
            bmp.setPixel( x, y, color.r, color.g, color.b );
        }
    }
    bmp.save( fileName );
    std::cout << "Image saved to " << fileName << " succesfully!" << std::endl;
}

//rat // table // book // sandwich // telega

int main( int argc, char* argv[] ) {
    setenv("OMP_PROC_BIND", "spread", 1);
    setenv("OMP_PLACES", "threads", 1);
    Kokkos::initialize(argc, argv); {
    srand(time( nullptr ));
    RayTracer* rayTracer = nullptr;
    ////OPTIONS

    ////RESOLUTION
    //int w = 8 ; int h = 5;
    //int w = 240 ; int h = 150;
    //int w = 640 ; int h = 400; //53 sec //
    //int w = 960 ; int h = 600;
    //int w = 1920 ; int h = 1200;
    int w = 3200; int h = 2000;

    ////NUM SAMPLES
    int depth = 2;
    int ambientSamples = 5;
    int lightSamples = 1;

// room scene ( 960x600 ) - 18.1 / 15.5 / 9.7 / 9.3 / 7.3
// room scene ( 3200x2000 ) - idk / 95 /
// rat scene ( 3200x2000 ) - 100 / 79 / 4.6
    auto start = std::chrono::high_resolution_clock::now();
    //sphereScene( rayTracer, w, h, depth, ambientSamples, lightSamples );//
    netRoomScene( rayTracer, w, h, depth, ambientSamples, lightSamples );//57 sec // 13.6 sec
    //simpleRoomScene( rayTracer, w, h, depth, ambientSamples, lightSamples );//57 sec // 13.6 sec
    //roomScene( rayTracer, w, h, depth, ambientSamples, lightSamples );//57 sec // 13.6 sec
    //ratScene( rayTracer, w, h, depth, ambientSamples, lightSamples );//2.3 sec // 1.7 sec // 8.67 sec
    //tableScene( rayTracer, w, h, depth, ambientSamples, lightSamples );//23 sec // 1.56 sec // 7,62 sec
    //bookScene( rayTracer, w, h, depth, ambientSamples, lightSamples );//130 sec // 31 sec //
    //sandwichScene( rayTracer, w, h, depth, ambientSamples, lightSamples );//3.29 sec //2 sec
    //cartScene( rayTracer, w, h, depth, ambientSamples, lightSamples );//118 sec // 1.96 sec
    //sksScene( rayTracer, w, h, depth, ambientSamples, lightSamples );//182 sec //1.6 sec
    //dogScene( rayTracer, w, h, depth, ambientSamples, lightSamples );//10 sec //9 sec
    //planeScene( rayTracer, w, h, depth, ambientSamples, lightSamples );//357 sec// 8 sec
    //modelRoomScene( rayTracer, w, h, depth, ambientSamples, lightSamples );//357 sec// 8 sec
    //carScene( rayTracer, w, h, depth, ambientSamples, lightSamples );//357 sec// 8 sec
    //cottageScene( rayTracer, w, h, depth, ambientSamples, lightSamples );//357 sec// 8 sec
    //hardScene( rayTracer, w, h, depth, ambientSamples, lightSamples ); //720 sec// 4 sec
    auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> loadTime = end - start;
    std::cout << "Model loads "<< loadTime.count() << " seconds" << std::endl;
    start = std::chrono::high_resolution_clock::now();;
    rayTracer->traceAllRays( RayTracer::PARALLEL );
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> renderTime = end - start;
    std::cout << "RayTracer works "<< renderTime.count() << " seconds" << std::endl;

    saveToBMP( rayTracer->getCanvas(), "out.bmp" );

    Denoiser denoiser( rayTracer->getCanvas() );
    denoiser.denoise();
    saveToBMP( rayTracer->getCanvas(), "outDenoised.bmp" );

    } Kokkos::finalize();
   //delete
   //TODO mb need init rayTracer more
   //TODO think about camera, i think its bad right now
    return 0;
}


// NET ROOM //

//(release) testing on 3200x2000, depth - 2, samples - 5, light - samples - 5, time - 610 sec

//(release) testing on 3200x2000, depth - 2, samples - 5, light - samples - 5, time - 252 sec


//(release) Kokkos CPU testing on 3200x2000, depth - 2, samples - 5, light - samples - 5, time - 310 sec

//(release) Kokkos CPU testing on 1920x1200, depth - 2, samples - 5, light - samples - 5, time - 110 sec


// END //

// TODO //
//поправить тень, один раз выбрать точки рандомные

// END //