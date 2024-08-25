#include <iostream>
#include "RayTracer.h"
#include "CubeMesh.h"
#include "PointLight.h"
#include "SpotLight.h"
#include "cstdlib"
#include "Denoiser.h"
//162.786 2 2 5 - 3200
// 3 sec - 960 2 2 2


void loadScene(Scene* scene, Vector <Mesh*>& meshes, Vector<Light*>& lights ) {
    for ( const auto mesh: meshes ) {
        scene->add( mesh );
    }
    for ( const auto light: lights ) {
        scene->add( light );
    }
}
void sphereScene( RayTracer*& rayTracer, int w, int h, int d, int numAS, int numLS ) {
    Camera* cam = new Camera( Vector3f(0, 0,-10000 ), Vector3f(0,0,1), 6000,3200,2000 );
    Scene* scene = new Scene();
    Canvas* canvas = new Canvas(w, h );

    Vector<Mesh*> meshes;
    Vector<Sphere*> spheres;
    Vector<Light*> lights;


    spheres.push_back(new Sphere(1500, Vector3f(0, 0, 3000), {YELLOW, -1, 0 }));

    spheres.push_back(new Sphere(300, Vector3f(2121, 0, 2250), {RED, -1 , 0 }));

    spheres.push_back(new Sphere(300, Vector3f(1030, 0, 1000),{GREEN, -1 , 0 }));

    spheres.push_back(new Sphere(300, Vector3f(-2121, 0, 2250),{PINK, -1, 0 }));

    spheres.push_back(new Sphere(300, Vector3f(-1030, 0, 1000),{CYAN, 500 }));


//    lights.push_back( new PointLight( Vector3f(2000,0,2900 ), 1200 ));
//    lights.push_back( new PointLight( Vector3f(-3500,0,0 ), 9999999 ));
//    lights.push_back( new PointLight( Vector3f(-1000,0,0 ), 500 ));
    loadScene( scene, meshes, lights );
    for ( auto& sphere: spheres ) {
        scene->add( *sphere );
    }
    rayTracer = new RayTracer( cam, scene, canvas, d, numAS, numLS );
}

void sphereScene1( RayTracer*& rayTracer, int w, int h, int d, int numAS, int numLS ) {
    Camera* cam = new Camera( Vector3f(0,10,0 ), Vector3f(0,0,1), 2000,3200,2000 );
    Scene* scene = new Scene();
    Canvas* canvas = new Canvas(w, h );

    Vector<Mesh*> meshes;
    Vector<Sphere*> spheres;
    Vector<Light*> lights;
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

////RAND SPHERE
    spheres.push_back( new Sphere(25, Vector3f(0, -10, 150), {BLUE, -1, 0 }) );
////LIGHTS
    spheres.push_back( new Sphere(10, Vector3f(0, 55, 150), {BLUE, 0.7}) );
    //lights.push_back( new PointLight( Vector3f(0,65,150), 0.55));
    int lightWidth = 20;
    //lights.push_back( new SpotLight( Vector3f(0 - lightWidth,65,180 - lightWidth), Vector3f(0 + lightWidth,65,180 + lightWidth), 0.7));
    //lights.push_back( new SpotLight( Vector3f(0 - lightWidth,-45,180 - lightWidth), Vector3f(0 + lightWidth,-45,180 + lightWidth), 0.7));


    for ( auto sphere: spheres ) {
        scene->add( *sphere );
    }
    loadScene( scene, meshes, lights );
    rayTracer = new RayTracer( cam, scene, canvas, d, numAS, numLS );
}

void netRoomScene( RayTracer*& rayTracer, int w, int h, int d, int numAS, int numLS ) {
    float FOV = 67.38;
    float dV = w / 2 / tan( FOV * M_PI / 180 / 2 );
    Camera* cam = new Camera( Vector3f(0,10,0 ), Vector3f(0,0,1), dV,w,h );
    //Camera* cam = new Camera( Vector3f(0,0,0 ), Vector3f(0,0,1), 6000,3200,2000 );
    Scene* scene = new Scene();
    Canvas* canvas = new Canvas( w, h );

    Vector<Mesh*> meshes;
    Vector<Light*> lights;
    float roomRoughness = 1;
////right
    meshes.push_back( new CubeMesh( Vector3f(70, -50, 0), Vector3f(80, 70, 600),
                                    { GREEN, -1 , roomRoughness } ) );
////left
    meshes.push_back(new CubeMesh( Vector3f(-80, -50, 0), Vector3f(-70, 70, 600),
                                   { RED, -1 , roomRoughness } ) );
////front
    meshes.push_back(new CubeMesh( Vector3f(-100, -50, 290), Vector3f(100, 70, 300),
                                   { GRAY, -1, roomRoughness } ) );
////back
    meshes.push_back(new CubeMesh( Vector3f(-100, -50, -10), Vector3f(100, 70, 0),
                                   { GRAY, -1 , roomRoughness } ) );
////down
    meshes.push_back(new CubeMesh( Vector3f(-100, -70, 0), Vector3f(100, -50, 620),
                                   { GRAY, -1 , roomRoughness } ) );
////up
    meshes.push_back(new CubeMesh( Vector3f(-100, 70, 0), Vector3f(100, 90, 620),
                                   { GRAY, -1 , roomRoughness } ) );

////RAND BLOCK
    auto* randBlockForward = new CubeMesh( Vector3f(-15, -50, 310), Vector3f(15, -30, 340) );
    randBlockForward->moveTo( Vector3f(0, -40, 325) );
    randBlockForward->scaleTo( Vector3f(30,100,30) );
    randBlockForward->rotate( Vector3f( 0,1,0), 25);
    randBlockForward->move( Vector3f(30,0,0));
//    randBlockForward->setMaterial({GRAY, 1});
    randBlockForward->setMaterial({GRAY, -1 , 1});
    randBlockForward->move( Vector3f(-10,0,-150));
    //randBlockForward->scaleTo( 200 );
    meshes.push_back(randBlockForward );

    auto* randBlockForward2 = new CubeMesh( Vector3f(-15, -50, 310), Vector3f(15, -30, 340) );
    randBlockForward2->moveTo( Vector3f(0, -40, 325) );
    randBlockForward2->scaleTo( Vector3f(30,260,30) );
    randBlockForward2->rotate( Vector3f( 0,1,0), -25);
    randBlockForward2->move( Vector3f(35,0,0));
    randBlockForward2->setMaterial({DARK_BLUE, -1, 0.1});
    randBlockForward2->move( Vector3f(-50,0,-100));
    //randBlockForward2->scaleTo( 200 );
    meshes.push_back(randBlockForward2 );

////LIGHTS

    //lights.push_back( new PointLight( Vector3f(0,65,150), 0.55));
    int lightWidth = 20;
    lights.push_back( new SpotLight( Vector3f(0 - lightWidth,65,180 - lightWidth), Vector3f(0 + lightWidth,65,180 + lightWidth), 0.8));

////LOADING...
    loadScene( scene, meshes, lights );
    rayTracer = new RayTracer( cam, scene, canvas, d, numAS, numLS );
}

void simpleRoomScene( RayTracer*& rayTracer, int w, int h, int d, int numAS, int numLS ) {
    Camera* cam = new Camera( Vector3f(0,0,300 ), Vector3f(0,0,1), 2400,3200,2000 );
    //Camera* cam = new Camera( Vector3f(0,0,0 ), Vector3f(0,0,1), 6000,3200,2000 );
    Scene* scene = new Scene();
    Canvas* canvas = new Canvas( w, h );

    Vector<Mesh*> meshes;
    Vector<Light*> lights;
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

    Vector<Mesh*> meshes;
    Vector<Light*> lights;
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
    auto* dog = new Mesh();
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
    auto* sks = new Mesh();
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
    auto* table = new Mesh();
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

    auto* plane = new Mesh();
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
    Vector<Mesh*> meshes;
    Vector<Light*> lights;

    auto* rat = new Mesh();
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
    Vector<Mesh*> meshes;
    Vector<Light*> lights;

    auto* table = new Mesh();
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
    Vector<Mesh*> meshes;
    Vector<Light*> lights;

    auto* book = new Mesh();
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
    Vector<Mesh*> meshes;
    Vector<Light*> lights;

    auto* sandwich = new Mesh();
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
    Vector<Mesh*> meshes;
    Vector<Light*> lights;

    auto* cart = new Mesh();
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
    Vector<Mesh*> meshes;
    Vector<Light*> lights;

    auto* sks = new Mesh();
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
    Vector<Mesh*> meshes;
    Vector<Light*> lights;

    auto* dog = new Mesh();
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
    Vector<Mesh*> meshes;
    Vector<Light*> lights;

    auto* plane = new Mesh();
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
    Vector<Mesh*> meshes;
    Vector<Light*> lights;

    auto* room = new Mesh();
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
    Vector<Mesh*> meshes;
    Vector<Light*> lights;

    auto* cottage = new Mesh();
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
    Vector<Mesh*> meshes;
    Vector<Light*> lights;

    auto* car = new Mesh();
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
    Vector<Mesh*> meshes;
    Vector<Light*> lights;
    auto* table = new Mesh();
    table->loadMesh( "/home/auser/dev/src/Collection/Models/table/model.obj" );
    //table->rotate( Vector3f( 0, 0, 1), 20 );
    //table->rotate( Vector3f( 1,0,0),20);
    table->rotate( Vector3f( 0,1,0),-120);
    table->move( Vector3f( 40,40,1200) );
    table->setMaterial( { RGB( 130, 130, 130 ), 1 , 0 } );



    auto* sks = new Mesh();
    sks->loadMesh( "/home/auser/dev/src/Collection/Models/sks/model.obj" );
    sks->rotate( Vector3f( 0, 0, 1), 0 );
    sks->rotate( Vector3f( 1,0,0),0);
    sks->rotate( Vector3f( 0,1,0),0);
    sks->move( Vector3f( -25,19,1080) );
    sks->setMaterial( { RGB( 130, 130, 130 ), 1 , 0 } );

    meshes.push_back( table );
    meshes.push_back( sks );


    lights.push_back( new PointLight( Vector3f(20 ,0,0), 0.5));
    loadScene( scene, meshes, lights );
    rayTracer = new RayTracer( cam, scene, canvas, d, numAS, numLS );
}

void audiScene( RayTracer*& rayTracer, int w, int h, int d, int numAS, int numLS ) {
    float FOV = 67.38;
    float dV = w / 2 / tan( FOV * M_PI / 180 / 2 );
    Camera* cam = new Camera( Vector3f(0,10,0 ), Vector3f(0,0,1), dV,w,h );
    //Camera* cam = new Camera( Vector3f(0,0,0 ), Vector3f(0,0,1), 6000,3200,2000 );
    Scene* scene = new Scene();
    Canvas* canvas = new Canvas( w, h );

    Vector<Mesh*> meshes;
    Vector<Light*> lights;
    float roomRefl = 0;
//////right
//    meshes.push_back( new CubeMesh( Vector3f(70, -50, 0), Vector3f(80, 70, 600),
//                                    { GREEN, -1 , roomRefl } ) );
//////left
//    meshes.push_back(new CubeMesh( Vector3f(-80, -50, 0), Vector3f(-70, 70, 600),
//                                   { RED, -1 , roomRefl } ) );
//////front
//    meshes.push_back(new CubeMesh( Vector3f(-100, -50, 290), Vector3f(100, 70, 300),
//                                   { GRAY, -1, roomRefl } ) );
//////back
//    meshes.push_back(new CubeMesh( Vector3f(-100, -50, -10), Vector3f(100, 70, 0),
//                                   { GRAY, -1 , roomRefl } ) );
//////down
//    meshes.push_back(new CubeMesh( Vector3f(-100, -70, 0), Vector3f(100, -50, 620),
//                                   { GRAY, -1 , roomRefl } ) );
//////up
//    meshes.push_back(new CubeMesh( Vector3f(-100, 70, 0), Vector3f(100, 90, 620),
//                                   { GRAY, -1 , roomRefl } ) );

////AUDI
    int roomLength = 300;
    int roomHeight = 140;
    auto* audi = new Mesh();
    audi->loadMesh( "/home/auser/dev/src/Collection/Models/audi/audi.obj" );
    audi->scaleTo( 100 );
    audi->moveTo(Vector3f( 0,0, roomLength / 2  ) );
////    //rat->rotate( Vector3f( 0, 0, 1), 45 );
    audi->rotate( Vector3f( 1,0,0),270);
    audi->rotate( Vector3f( 0,1,0), 145);
    audi->move( Vector3f( -4,0,3) );
    audi->setMinPoint( Vector3f( 0,-50,0), 1 );
    audi->setMaterial( { BLUE, -1 , 0 } );

    ////Grill
   // Material grill = { BLACK, 0.8, 0.3/*0.3*/, 0.1 };
    Material grill = { BLACK, -1, 0.3 };
    //Front
    //Top of grill
//    audi->setMaterial( grill, 47 );
//    audi->setMaterial( grill, 38 );
//    audi->setMaterial( grill, 25 );
//    //Left/Right grill
//    audi->setMaterial( grill, 34 );
//    //Bottom of grill
//    audi->setMaterial( grill, 31 );
//    audi->setMaterial( grill, 58 );
//    //Boxes
//    audi->setMaterial( grill, 27 ); //front boxes( grill )
//    audi->setMaterial( grill, 30 ); //front boxes( grill )
//    //Radiator
//    Material radiator = { { 100, 100, 100 }, 1, 0, 1 };
//    audi->setMaterial( radiator, 59 ); //radiator bottom
//    audi->setMaterial( radiator, 57 ); //l/r radiator
//    audi->setMaterial( radiator, 60 ); //radiator front
//    //Back
//    audi->setMaterial( grill, 33 );
//    audi->setMaterial( radiator, 73 );
//
//
//    ////Signs
//    Material signs = { { 220, 220, 220 }, 1, 0.7, 1 };
//    //Front
//    audi->setMaterial( signs, 1 );
//    //Back
//    audi->setMaterial( signs, 23 ); // audi
//    audi->setMaterial( signs, 39 );
//
//    ////Wheels
//    ////Tires
//    Material tires = { BLACK, 1 , 0, 1 };
//    //Front
//    audi->setMaterial( tires, 5 ); //front tire
//    //Back
//    audi->setMaterial( tires, 20 ); //back tires
//    ////Rims
//    Material rims = { { 220, 220, 220 }, 1, 0.4, 1 };
//    //Front
//    audi->setMaterial( rims, 15 );
//    audi->setMaterial( rims, 13 );
//    audi->setMaterial( rims, 14 );
//    //Back
//    audi->setMaterial( rims, 22 );
//    audi->setMaterial( rims, 21 );
//    ////Brakes
//    Material mainBreaks = { RED, 1 , 0, 1 };
//    audi->setMaterial( mainBreaks, 134 ); //main tormoza(red) back
//    audi->setMaterial( mainBreaks, 130 ); //main tormoza(red) front
//    audi->setMaterial( mainBreaks, 41 ); //hueta na tormoze
//    audi->setMaterial( mainBreaks, 42 ); //hueta na tormoze
//    Material secondBreaks = { GRAY, 1 , 0, 1 };
//    audi->setMaterial( secondBreaks, 132 ); //tormoz disk big back
//    audi->setMaterial( secondBreaks, 82 ); //tormoz disk big
//    audi->setMaterial( secondBreaks, 12 ); //tormoz disk
//    audi->setMaterial( secondBreaks, 18 ); //tormoz disk
//
//    ////Bolts
//    Material bolts = { YELLOW, 1, 0.2, 1 };
//    audi->setMaterial( bolts, 9 );
//    audi->setMaterial( bolts, 10 );
//    audi->setMaterial( bolts, 8 ); //big bolt
//    audi->setMaterial( bolts, 16 ); //BOLTI back
//    //Sign
//    Material bSign = signs;
//    audi->setMaterial( bSign, 6 ); //SIGN
//    audi->setMaterial( { BLACK, 1 , 0, 1 }, 2 ); //sign back wheel
//    //Obodok
//    Material obod = { YELLOW, 1, 0.2, 1 };
//    audi->setMaterial( obod, 7 ); //obodok
//    audi->setMaterial( obod, 4 ); //obodok back wheel
//
//
//    ////BODY
//    Material body = { DARK_BLUE, 1 , 0/*0.15*/, 1 };
//    audi->setMaterial( body, 0 ); //sign podstavka
//    audi->setMaterial( body, 44 ); //back bagajnik
//    audi->setMaterial( body, 146 ); //l/r door front
//    audi->setMaterial( body, 145 ); //back bumper
//    audi->setMaterial( body, 144 ); //front krilo
//    audi->setMaterial( body, 138 ); //hueta mezhdu l/r windows
//    audi->setMaterial( body, 139 );//l/r porogi
//    audi->setMaterial( body, 136 ); //l/r porogi
//    audi->setMaterial( body, 80 ); //back door
//    audi->setMaterial( body, 66 ); //capot
//    audi->setMaterial( body, 46 ); //roof and back
//    audi->setMaterial( body, 49 ); // front bumper
//    audi->setMaterial( body, 24 ); //side mirrors
//    audi->setMaterial( body, 157 ); // back mid hz
//    audi->setMaterial( body, 29 ); //side mirrors mid
//    audi->setMaterial( body, 35 ); //side mirrors mid
//    audi->setMaterial( body, 53 ); // spoiler
//    audi->setMaterial( body, 50 ); //FRONT
//    audi->setMaterial( body, 56 ); //FRONT
//    audi->setMaterial( body, 128 );//okolo grill
//    audi->setMaterial( body, 51 );//obvodka u grill
//    audi->setMaterial( body, 54 );// top grill obvodka
//    audi->setMaterial( body, 52 ); //obvodka u kapota
//    audi->setMaterial( body, 103 ); // okolo tires
//    audi->setMaterial( body, 61 ); //l/r grill obdodka
//    audi->setMaterial( body, 79 ); // to kosmos pimpo4ka
//    audi->setMaterial( body, 78 ); // roof hueta
//    audi->setMaterial( body, 76 ); // roof hueta
//    audi->setMaterial( body, 63 ); // obvodka back mirror
//    audi->setMaterial( body, 72 ); //back bumper obvodka
//    audi->setMaterial( body, 75 ); //back bumper obvodka
//    audi->setMaterial( body, 74 ); //back bumper obvodka( ele vidno)
//    audi->setMaterial( body, 28 ); //obodok right mirror
//    audi->setMaterial( body, 83 ); //back
//    audi->setMaterial( body, 55 ); //front obodot snuzu
//    ////Obvodka
//    Material obvodka = { BLACK, 0.7 , 0, 0.5 };
//    audi->setMaterial( obvodka, 150 ); //l/r windows obvodka
//    audi->setMaterial( obvodka, 70 ); //front window obvodka l/r
//    audi->setMaterial( obvodka, 68 ); //front window obvodka bottom
//    audi->setMaterial( obvodka, 65 ); // obvodka side mirrors (doors)
//    audi->setMaterial( obvodka, 148 ); //obvodka door front side
//
//    ////Handles
//    Material handles = body;
//    audi->setMaterial( handles, 37 );
//    audi->setMaterial( handles, 36 );
//
//    ////Lights
//    //Front
//    Material mLights = {BLACK, 1 , 0.2, 0 };
//    audi->setMaterial( mLights, 153 );
//    //Back
//    audi->setMaterial( mLights, 114 );
//    audi->setMaterial( mLights, 112 );
//
//    ////Windows
//    audi->setMaterial( { BLACK, 1 , 0.2, 1 }, 32 ); //l/r mirror
//    audi->setMaterial( { BLACK, 1 , 0.2, 1 }, 71 ); //back mirror
//    audi->setMaterial( { BLACK, 1 , 0.2, 1 }, 67 ); //front mirror
//    audi->setMaterial( { BLACK, 1 , 0.2, 1 }, 45 ); //side mirrors ( doors )
//
//    ////Other
//    audi->setMaterial( {{200,200,200}, 1 , 0.6, 1 }, 3 ); //vihlop
    meshes.push_back( audi );

////LIGHTS

//    lights.push_back( new PointLight( Vector3f(0,65,150), 0.55));
    Vector<Sphere*> spheres;
    spheres.push_back( new Sphere( 10, Vector3f(0, 50, 200), {WHITE, 1 } ) );
    for ( auto sphere: spheres ){
        scene->add( *sphere );
    }
////LOADING...
    loadScene( scene, meshes, lights );
    rayTracer = new RayTracer( cam, scene, canvas, d, numAS, numLS );
}

void sphereRoomScene( RayTracer*& rayTracer, int w, int h, int d, int numAS, int numLS ) {
    float FOV = 67.38;
    float dV = w / 2 / tan( FOV * M_PI / 180 / 2 );
    Camera* cam = new Camera( Vector3f(0,10,0 ), Vector3f(0,0,1), dV,w,h );
    //Camera* cam = new Camera( Vector3f(0,0,0 ), Vector3f(0,0,1), 6000,3200,2000 );
    Scene* scene = new Scene();
    Canvas* canvas = new Canvas( w, h );

    Vector<Mesh*> meshes;
    Vector<Light*> lights;
    float roomRefl = 1;

    Material floor = {GRAY, -1, 0 };
    floor.setTexture( "/home/auser/dev/src/Collection/Textures/WoodFloorBright/");
    Material wall = {GRAY, -1, 0 };
    wall.setTexture( "/home/auser/dev/src/Collection/Textures/Plaster/");
    Material ceil = {GRAY, -1, 0 };
    ceil.setTexture( "/home/auser/dev/src/Collection/Textures/PorceLain/");
    Material ground = {GRAY, -1, 0 };
    ground.setTexture( "/home/auser/dev/src/Collection/Textures/Ground/");
    Material carpet = {GRAY, -1, 0 };
    carpet.setTexture( "/home/auser/dev/src/Collection/Textures/Carpet/");
    Material giraffe = {GRAY, -1, 0 };
    giraffe.setTexture( "/home/auser/dev/src/Collection/Textures/GiraffeFur/");
    Material mink = {GRAY, -1, 0 };
    mink.setTexture( "/home/auser/dev/src/Collection/Textures/MinkFur/");
    Material marble = {GRAY, -1, 0 };
    marble.setTexture( "/home/auser/dev/src/Collection/Textures/Marble/");

////right
    meshes.push_back( new CubeMesh( Vector3f(70, -50, 0), Vector3f(80, 70, 600),
                                   wall ) );
////left
    meshes.push_back(new CubeMesh( Vector3f(-80, -50, 0), Vector3f(-70, 70, 600),
                                   wall ) );
////front
    meshes.push_back(new CubeMesh( Vector3f(-100, -50, 290), Vector3f(100, 70, 300),
                                   wall ) );
////back
    meshes.push_back(new CubeMesh( Vector3f(-100, -50, -10), Vector3f(100, 70, 0),
                                   wall ) );
////floor
    meshes.push_back(new CubeMesh( Vector3f(-100, -70, 0), Vector3f(100, -50, 620),
                                   floor ) );
////ceil
    meshes.push_back(new CubeMesh( Vector3f(-100, 70, 0), Vector3f(100, 90, 620),
                                   wall ) );


    ////RAND BLOCK
    auto* randBlockForward = new CubeMesh( Vector3f(-15, -50, 310), Vector3f(15, -30, 340) );
    randBlockForward->moveTo( Vector3f(20, -40, 175) );
    randBlockForward->scaleTo( Vector3f(30,100,30) );
    randBlockForward->rotate( Vector3f( 0,1,0), 25);
    randBlockForward->setMaterial( carpet );
    meshes.push_back(randBlockForward );

    auto* randBlockForward2 = new CubeMesh( Vector3f(-15, -50, 310), Vector3f(15, -30, 340) );
    randBlockForward2->moveTo( Vector3f(-35, -40, 205) );
    randBlockForward2->scaleTo( Vector3f(30,260,30) );
    randBlockForward2->rotate( Vector3f( 0,1,0), 45);
    randBlockForward2->setMaterial( ceil );
    meshes.push_back(randBlockForward2 );

////Spheres
    Vector<Sphere* > spheres;
    spheres.push_back( new Sphere( 20, Vector3f(20, 0, 175), marble ) );


////LIGHTS

//    lights.push_back( new PointLight( Vector3f(0,65,150), 0.55));
    int lightWidth = 20;
    meshes.push_back( new CubeMesh( Vector3f(0 - lightWidth,64,150 - lightWidth), Vector3f(0 + lightWidth,65,150 + lightWidth), { WHITE, 1.2 }));

////LOADING...
    for ( auto sphere: spheres ) {
        scene->add( *sphere );
    }
    loadScene( scene, meshes, lights );
    rayTracer = new RayTracer( cam, scene, canvas, d, numAS, numLS );
}


void dragonScene( RayTracer*& rayTracer, int w, int h, int d, int numAS, int numLS ) {
    float FOV = 67.38;
    float dV = w / 2 / tan( FOV * M_PI / 180 / 2 );
    Camera* cam = new Camera( Vector3f(0,10,0 ), Vector3f(0,0,1), dV,w,h );
    //Camera* cam = new Camera( Vector3f(0,0,0 ), Vector3f(0,0,1), 6000,3200,2000 );
    Scene* scene = new Scene();
    Canvas* canvas = new Canvas( w, h );

    Vector<Mesh*> meshes;
    Vector<Light*> lights;
    float roomRefl = 0;
////right
    meshes.push_back( new CubeMesh( Vector3f(70, -50, 0), Vector3f(80, 70, 600),
                                    { GREEN, -1 , roomRefl } ) );
////left
    meshes.push_back(new CubeMesh( Vector3f(-80, -50, 0), Vector3f(-70, 70, 600),
                                   { RED, -1 , roomRefl } ) );
////front
    meshes.push_back(new CubeMesh( Vector3f(-100, -50, 290), Vector3f(100, 70, 300),
                                   { GRAY, -1, roomRefl } ) );
////back
    meshes.push_back(new CubeMesh( Vector3f(-100, -50, -10), Vector3f(100, 70, 0),
                                   { GRAY, -1 , roomRefl } ) );
////down
    meshes.push_back(new CubeMesh( Vector3f(-100, -70, 0), Vector3f(100, -50, 620),
                                   { GRAY, -1 , roomRefl } ) );
////up
    meshes.push_back(new CubeMesh( Vector3f(-100, 70, 0), Vector3f(100, 90, 620),
                                   { GRAY, -1 , roomRefl } ) );

////RAND BLOCK
    auto* dragon = new Mesh();
    dragon->loadMesh( "/home/auser/dev/src/Collection/Models/dragon/armadillo.obj" );
    dragon->setMaterial(  { GRAY, -1 , roomRefl } );
    dragon->scaleTo(100 );
    dragon->moveTo( {0,0,150} );
    dragon->rotate( Vector3f( 1,0,0),-30);
    dragon->setMinPoint( {0,-50,0}, 1 );
    meshes.push_back( dragon );
////LIGHTS

    lights.push_back( new PointLight( Vector3f(0,65,150), 0.6));
    int lightWidth = 20;
    //lights.push_back( new SpotLight( Vector3f(0 - lightWidth,65,180 - lightWidth), Vector3f(0 + lightWidth,65,180 + lightWidth), 0.7));

////LOADING...
    loadScene( scene, meshes, lights );
    rayTracer = new RayTracer( cam, scene, canvas, d, numAS, numLS );
}


void testScene( RayTracer*& rayTracer, int w, int h, int d, int numAS, int numLS ) {
    float FOV = 67.38;
    float dV = w / 2 / tan( FOV * M_PI / 180 / 2 );
    Camera* cam = new Camera( Vector3f(0,10,0 ), Vector3f(0,0,1), dV,w,h );
    //Camera* cam = new Camera( Vector3f(0,0,0 ), Vector3f(0,0,1), 6000,3200,2000 );
    Scene* scene = new Scene();
    Canvas* canvas = new Canvas( w, h );

    Vector<Mesh*> meshes;
    Vector<Light*> lights;
    float roomRefl = 1;

    Material floor = {GRAY, -1, 0 };
    floor.setTexture( "/home/auser/dev/src/Collection/Textures/WoodFloorBright/");

////right
    meshes.push_back( new CubeMesh( Vector3f(70, -50, 0), Vector3f(80, 70, 600),
                                    floor ) );
////left
    meshes.push_back(new CubeMesh( Vector3f(-80, -50, 0), Vector3f(-70, 70, 600),
                                   floor ) );
////front
    meshes.push_back(new CubeMesh( Vector3f(-100, -50, 290), Vector3f(100, 70, 300),
                                   floor ) );
////back
    meshes.push_back(new CubeMesh( Vector3f(-100, -50, -10), Vector3f(100, 70, 0),
                                   floor ) );
////floor
    meshes.push_back(new CubeMesh( Vector3f(-100, -70, 0), Vector3f(100, -50, 620),
                                   floor ) );
////ceil
    meshes.push_back(new CubeMesh( Vector3f(-100, 70, 0), Vector3f(100, 90, 620),
                                   floor ) );


    ////RAND BLOCK
    auto* randBlockForward = new CubeMesh( Vector3f(0, 0, 0), Vector3f(30, 30, 30) );
    randBlockForward->moveTo( Vector3f(15, -10, 210 - 60) );
    randBlockForward->setMaterial( floor );
    meshes.push_back(randBlockForward );

    auto* randBlockForward1 = new CubeMesh( Vector3f(0, 0, 0), Vector3f(60, 60, 60) );
    randBlockForward1->moveTo( Vector3f(-30, -10, 225 - 60) );
    randBlockForward1->setMaterial( floor );
    meshes.push_back(randBlockForward1 );

////LIGHTS

    lights.push_back( new PointLight( Vector3f(0,0,0), 3));
    lights.push_back( new PointLight( Vector3f(0,0,290), 3));
    lights.push_back( new PointLight( Vector3f(0,65,250), 3));
//    int lightWidth = 20;
//    meshes.push_back( new CubeMesh( Vector3f(0 - lightWidth,64,150 - lightWidth), Vector3f(0 + lightWidth,65,150 + lightWidth), { WHITE, 1.2 }));

    loadScene( scene, meshes, lights );
    rayTracer = new RayTracer( cam, scene, canvas, d, numAS, numLS );
}

int main( int argc, char* argv[] ) {
    setenv("OMP_PROC_BIND", "spread", 1);
    setenv("OMP_PLACES", "threads", 1);
    Kokkos::initialize(argc, argv); {
    std::cout << "Default Execution Space: " << Kokkos::DefaultExecutionSpace::name() << std::endl;
    srand(time( nullptr ));
    RayTracer* rayTracer = nullptr;
    ////OPTIONS

    ////RESOLUTION
    //int w = 8 ; int h = 5;
    //int w = 240 ; int h = 150;
    //int w = 640 ; int h = 400; //53 sec //
    //int w = 960 ; int h = 600; //3 sec
    //int w = 1920 ; int h = 1200;
    int w = 3200; int h = 2000;

    // 160 sec 2 5 2 - 3200
    // 29.5 sec 2 5 5 - 960
    ////NUM SAMPLES
    int depth = 3;
    int ambientSamples = 5;
    int lightSamples = 2;

// room scene ( 960x600 ) - 18.1 / 15.5 / 9.7 / 9.3 / 7.3
// room scene ( 3200x2000 ) - idk / 95 /
// rat scene ( 3200x2000 ) - 100 / 79 / 4.6
    auto start = std::chrono::high_resolution_clock::now();
    //sphereScene( rayTracer, w, h, depth, ambientSamples, lightSamples );//
    //sphereScene1( rayTracer, w, h, depth, ambientSamples, lightSamples );//
    //netRoomScene( rayTracer, w, h, depth, ambientSamples, lightSamples );//57 sec // 13.6 sec
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
    //audiScene( rayTracer, w, h, depth, ambientSamples, lightSamples ); //720 sec// 4 sec
    sphereRoomScene( rayTracer, w, h, depth, ambientSamples, lightSamples );//57 sec // 13.6 sec
    //dragonScene( rayTracer, w, h, depth, ambientSamples, lightSamples );//57 sec // 13.6 sec
    //testScene( rayTracer, w, h, depth, ambientSamples, lightSamples );//57 sec // 13.6 sec
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> loadTime = end - start;
    std::cout << "Model loads "<< loadTime.count() << " seconds" << std::endl;
    start = std::chrono::high_resolution_clock::now();;
    rayTracer->render( RayTracer::PARALLEL );
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> renderTime = end - start;
    std::cout << "RayTracer works "<< renderTime.count() << " seconds" << std::endl;

    rayTracer->getCanvas()->saveToPNG( "out.png" );

    Denoiser::denoise( rayTracer->getCanvas()->getColorData(), rayTracer->getCanvas()->getNormalData(),rayTracer->getCanvas()->getAlbedoData(), w, h );
    rayTracer->getCanvas()->saveToPNG( "outDenoised.png" );


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