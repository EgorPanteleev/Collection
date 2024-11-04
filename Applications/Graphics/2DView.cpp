#include <iostream>
#include "RayTracer.h"
#include "CubeMesh.h"
#include "PointLight.h"
#include "SpotLight.h"
#include "cstdlib"
#include "Denoiser.h"
#include "GroupOfMeshes.h"
#include "Triangles.h"
//162.786 2 2 5 - 3200
// 3 sec - 960 2 2 2


//void test() {
//    Triangles triangles;
//    Vec3d v1 = { 0, 0, 0 };
//    Vec3d v2 = { 0, 0, 1 };
//    Vec3d v3 = { 0, 1, 0 };
//    Vec3d v4 = { 0, 1, 0 };
//    Vec3d v5 = { 0, 1, 1 };
//    Vec3d v6 = { 1, 0, 0 };
//    triangles.addTriangle( v1, v2, v3 );
//    triangles.addTriangle( v4, v5, v6 );
//    std::cout << "Vertices size "<<triangles.vertices.size()<<std::endl;
//    std::cout<< "Indices size "<< triangles.indices.size()<<std::endl;
//    std::cout << "Indexes:" << std::endl;
//    for ( auto ind: triangles.indices ) {
//        std::cout << ind << std::endl;
//    }
//}

void loadScene(Scene* scene, Vector <Mesh*>& meshes, Vector<Light*>& lights ) {
    for ( const auto mesh: meshes ) {
        scene->add( mesh );
    }
    for ( const auto light: lights ) {
        scene->add( light );
    }
}
void sphereScene( RayTracer*& rayTracer, int w, int h, int d, int numAS, int numLS ) {
    Camera* cam = new Camera( Vec3d(0, 0,-10000 ), Vec3d(0,0,1), 6000,3200,2000 );
    Scene* scene = new Scene();
    Canvas* canvas = new Canvas(w, h );

    Vector<Mesh*> meshes;
    Vector<Sphere*> spheres;
    Vector<Light*> lights;


    spheres.push_back(new Sphere(1500, Vec3d(0, 0, 3000), {YELLOW, -1, 0 }));

    spheres.push_back(new Sphere(300, Vec3d(2121, 0, 2250), {RED, -1 , 0 }));

    spheres.push_back(new Sphere(300, Vec3d(1030, 0, 1000),{GREEN, -1 , 0 }));

    spheres.push_back(new Sphere(300, Vec3d(-2121, 0, 2250),{PINK, -1, 0 }));

    spheres.push_back(new Sphere(300, Vec3d(-1030, 0, 1000),{CYAN, 500 }));


//    lights.push_back( new PointLight( Vec3d(2000,0,2900 ), 1200 ));
//    lights.push_back( new PointLight( Vec3d(-3500,0,0 ), 9999999 ));
//    lights.push_back( new PointLight( Vec3d(-1000,0,0 ), 500 ));
    loadScene( scene, meshes, lights );
    for ( auto sphere: spheres ) {
        scene->add( sphere );
    }
    rayTracer = new RayTracer( cam, scene, canvas, d, numAS, numLS );
}

void sphereScene1( RayTracer*& rayTracer, int w, int h, int d, int numAS, int numLS ) {
    Camera* cam = new Camera( Vec3d(0,10,0 ), Vec3d(0,0,1), 2000,3200,2000 );
    Scene* scene = new Scene();
    Canvas* canvas = new Canvas(w, h );

    Vector<Mesh*> meshes;
    Vector<Sphere*> spheres;
    Vector<Light*> lights;
    double roomRefl = 0;
////right
    meshes.push_back( new CubeMesh( Vec3d(70, -50, 0), Vec3d(80, 70, 600),
                                    { GREEN, -1 , roomRefl } ) );
////left
    meshes.push_back(new CubeMesh( Vec3d(-80, -50, 0), Vec3d(-70, 70, 600),
                                   { RED, -1 , roomRefl } ) );
////front
    meshes.push_back(new CubeMesh( Vec3d(-100, -50, 290), Vec3d(100, 70, 300),
                                   { GRAY, -1 , roomRefl } ) );
////back
    meshes.push_back(new CubeMesh( Vec3d(-100, -50, -10), Vec3d(100, 70, 0),
                                   { GRAY, -1 , roomRefl } ) );
////down
    meshes.push_back(new CubeMesh( Vec3d(-100, -70, 0), Vec3d(100, -50, 620),
                                   { GRAY, -1 , roomRefl } ) );
////up
    meshes.push_back(new CubeMesh( Vec3d(-100, 70, roomRefl), Vec3d(100, 90, 620),
                                   { GRAY, -1 , 0 } ) );

////RAND SPHERE
    spheres.push_back( new Sphere(25, Vec3d(0, -10, 150), {BLUE, -1, 0 }) );
////LIGHTS
    spheres.push_back( new Sphere(10, Vec3d(0, 55, 150), {BLUE, 0.7}) );
    //lights.push_back( new PointLight( Vec3d(0,65,150), 0.55));
    int lightWidth = 20;
    //lights.push_back( new SpotLight( Vec3d(0 - lightWidth,65,180 - lightWidth), Vec3d(0 + lightWidth,65,180 + lightWidth), 0.7));
    //lights.push_back( new SpotLight( Vec3d(0 - lightWidth,-45,180 - lightWidth), Vec3d(0 + lightWidth,-45,180 + lightWidth), 0.7));


    for ( auto sphere: spheres ) {
        scene->add( sphere );
    }
    loadScene( scene, meshes, lights );
    rayTracer = new RayTracer( cam, scene, canvas, d, numAS, numLS );
}

void netRoomScene( RayTracer*& rayTracer, int w, int h, int d, int numAS, int numLS ) {
    double FOV = 67.38;
    double dV = w / 2 / tan( FOV * M_PI / 180 / 2 );
    Camera* cam = new Camera( Vec3d(0,10,0 ), Vec3d(0,0,1), dV,w,h );
    //Camera* cam = new Camera( Vec3d(0,0,0 ), Vec3d(0,0,1), 6000,3200,2000 );
    Scene* scene = new Scene();
    Canvas* canvas = new Canvas( w, h );

    Vector<Mesh*> meshes;
    Vector<Light*> lights;
    double roomRoughness = 1;
////right
    meshes.push_back( new CubeMesh( Vec3d(70, -50, 0), Vec3d(80, 70, 600),
                                    { GREEN, -1 , roomRoughness } ) );
////left
    meshes.push_back(new CubeMesh( Vec3d(-80, -50, 0), Vec3d(-70, 70, 600),
                                   { RED, -1 , roomRoughness } ) );
////front
    meshes.push_back(new CubeMesh( Vec3d(-100, -50, 290), Vec3d(100, 70, 300),
                                   { GRAY, -1, roomRoughness } ) );
////back
    meshes.push_back(new CubeMesh( Vec3d(-100, -50, -10), Vec3d(100, 70, 0),
                                   { GRAY, -1 , roomRoughness } ) );
////down
    meshes.push_back(new CubeMesh( Vec3d(-100, -70, 0), Vec3d(100, -50, 620),
                                   { GRAY, -1 , roomRoughness } ) );
////up
    meshes.push_back(new CubeMesh( Vec3d(-100, 70, 0), Vec3d(100, 90, 620),
                                   { GRAY, -1 , roomRoughness } ) );

////RAND BLOCK
    auto* randBlockForward = new CubeMesh( Vec3d(-15, -50, 310), Vec3d(15, -30, 340) );
    randBlockForward->moveTo( Vec3d(0, -40, 325) );
    randBlockForward->scaleTo( Vec3d(30,100,30) );
    randBlockForward->rotate( Vec3d( 0,1,0), 25);
    randBlockForward->move( Vec3d(30,0,0));
//    randBlockForward->setMaterial({GRAY, 1});
    randBlockForward->setMaterial({GRAY, -1 , 1});
    randBlockForward->move( Vec3d(-10,0,-150));
    //randBlockForward->scaleTo( 200 );
    meshes.push_back(randBlockForward );

    auto* randBlockForward2 = new CubeMesh( Vec3d(-15, -50, 310), Vec3d(15, -30, 340) );
    randBlockForward2->moveTo( Vec3d(0, -40, 325) );
    randBlockForward2->scaleTo( Vec3d(30,260,30) );
    randBlockForward2->rotate( Vec3d( 0,1,0), -25);
    randBlockForward2->move( Vec3d(35,0,0));
    randBlockForward2->setMaterial({DARK_BLUE, -1, 0.1});
    randBlockForward2->move( Vec3d(-50,0,-100));
    //randBlockForward2->scaleTo( 200 );
    meshes.push_back(randBlockForward2 );

////LIGHTS

    //lights.push_back( new PointLight( Vec3d(0,65,150), 0.55));
    int lightWidth = 20;
    lights.push_back( new SpotLight( Vec3d(0 - lightWidth,65,180 - lightWidth), Vec3d(0 + lightWidth,65,180 + lightWidth), 0.8));

////LOADING...
    loadScene( scene, meshes, lights );
    rayTracer = new RayTracer( cam, scene, canvas, d, numAS, numLS );
}

void simpleRoomScene( RayTracer*& rayTracer, int w, int h, int d, int numAS, int numLS ) {
    Camera* cam = new Camera( Vec3d(0,0,300 ), Vec3d(0,0,1), 2400,3200,2000 );
    //Camera* cam = new Camera( Vec3d(0,0,0 ), Vec3d(0,0,1), 6000,3200,2000 );
    Scene* scene = new Scene();
    Canvas* canvas = new Canvas( w, h );

    Vector<Mesh*> meshes;
    Vector<Light*> lights;
////right
    meshes.push_back( new CubeMesh( Vec3d(80, -50, 0), Vec3d(100, 50, 600),
                                    { GRAY, -1 , 0 } ) );
////left
    meshes.push_back(new CubeMesh( Vec3d(-100, -50, 0), Vec3d(-80, 50, 600),
                                   { GRAY, -1 , 0 } ) );
////front
    meshes.push_back(new CubeMesh( Vec3d(-100, -50, 600), Vec3d(100, 50, 610),
                                   { GRAY, -1 , 0 } ) );
////back
    meshes.push_back(new CubeMesh( Vec3d(-100, -50, -10), Vec3d(100, 50, 0),
                                   { GRAY, -1 , 0 } ) );
////down
    meshes.push_back(new CubeMesh( Vec3d(-100, -70, 0), Vec3d(100, -50, 620),
                                   { GRAY, -1 , 0 } ) );
////up
    meshes.push_back(new CubeMesh( Vec3d(-100, 50, 0), Vec3d(100, 70, 620),
                                   { GRAY, -1 , 0 } ) );

////RAND BLOCK
    auto* randBlockForward = new CubeMesh( Vec3d(-15, -50, 310), Vec3d(15, -30, 340) );
    randBlockForward->moveTo( Vec3d(0, -40, 325) );
    randBlockForward->scaleTo( Vec3d(20,90,20) );
    randBlockForward->rotate( Vec3d( 0,1,0), 45);
    randBlockForward->move( Vec3d(30,0,0));
    randBlockForward->setMaterial({RED, 1 , 0});
    randBlockForward->move( Vec3d(-50,0,180));
    //randBlockForward->scaleTo( 200 );
    meshes.push_back(randBlockForward );

////LIGHTS

    lights.push_back( new PointLight( Vec3d(0,25,450), 0.45));

//    lights.push_back( new PointLight( Vec3d(-75,35,595), 0.15));
//    lights.push_back( new PointLight( Vec3d(75,35,595), 0.15));
//    lights.push_back( new PointLight( Vec3d(-75,35,5), 0.15));
//    lights.push_back( new PointLight( Vec3d(75,35,5), 0.15));
////LOADING...
    loadScene( scene, meshes, lights );
    rayTracer = new RayTracer( cam, scene, canvas, d, numAS, numLS );
}


void roomScene( RayTracer*& rayTracer, int w, int h, int d, int numAS, int numLS ) {
    Camera* cam = new Camera( Vec3d(0,0,300 ), Vec3d(0,0,1), 3000,3200,2000 );
    //Camera* cam = new Camera( Vec3d(0,0,0 ), Vec3d(0,0,1), 6000,3200,2000 );
    Scene* scene = new Scene();
    Canvas* canvas = new Canvas( w, h );

    Vector<Mesh*> meshes;
    Vector<Light*> lights;
////right
    meshes.push_back( new CubeMesh( Vec3d(80, -50, 0), Vec3d(100, 50, 600),
                                    { GRAY, -1 , 0 } ) );
////left
    meshes.push_back(new CubeMesh( Vec3d(-100, -50, 0), Vec3d(-80, 50, 600),
                                   { GRAY, -1 , 0 } ) );
////front
    meshes.push_back(new CubeMesh( Vec3d(-100, -50, 600), Vec3d(100, 50, 610),
                                   { GRAY, -1 , 0 } ) );
////back
    meshes.push_back(new CubeMesh( Vec3d(-100, -50, -10), Vec3d(100, 50, 0),
                                   { GRAY, -1 , 0 } ) );
////down
    meshes.push_back(new CubeMesh( Vec3d(-100, -70, 0), Vec3d(100, -50, 620),
                                   { GRAY, -1 , 0 } ) );
////up
    meshes.push_back(new CubeMesh( Vec3d(-100, 50, 0), Vec3d(100, 70, 620),
                                   { GRAY, -1 , 0 } ) );

////RAND BLOCK
    auto* randBlockForward = new CubeMesh( Vec3d(-15, -50, 310), Vec3d(15, -30, 340) );
    randBlockForward->moveTo( Vec3d(0, -40, 325) );
    randBlockForward->scaleTo( Vec3d(20,90,20) );
    randBlockForward->rotate( Vec3d( 0,1,0), 45);
    randBlockForward->move( Vec3d(30,0,0));
    randBlockForward->setMaterial({RED, 1 , 0});
    randBlockForward->move( Vec3d(-95,0,200));
    //randBlockForward->scaleTo( 200 );
    meshes.push_back(randBlockForward );

////DOG
    auto* dog = new Mesh();
    dog->loadMesh( "/home/auser/dev/src/Collection/Models/dog/model.obj" );
    //dog->rotate( Vec3d( 0, 0, 1), 45 );
    dog->rotate( Vec3d( 1,0,0),-90);
    dog->rotate( Vec3d( 0,1,0),-235);
    dog->scaleTo( 50 );
    dog->move( Vec3d( -10,0,520) );
    dog->setMinPoint( { 0, -50, 0 }, 1 );
    dog->setMaterial( { GRAY, 1 , 0 } );
    meshes.push_back( dog );
    ////SKS
    auto* sks = new Mesh();
    sks->loadMesh( "/home/auser/dev/src/Collection/Models/sks/model.obj" );
    sks->rotate( Vec3d( 0, 0, 1), 60 );
    //sks->rotate( Vec3d( 1,0,0),100);
    sks->rotate( Vec3d( 0,1,0), -120);
    sks->scaleTo(45);
    sks->setMinPoint( { 0, -50, 0 }, 1 );
    sks->move( Vec3d( 39,0,305) );
    sks->move( Vec3d(-95,0,200));
    sks->setMaterial( { GRAY, 1 , 0 } );
    meshes.push_back( sks );

    ////TABLE
    auto* table = new Mesh();
    table->loadMesh( "/home/auser/dev/src/Collection/Models/table/model.obj" );
    //table->rotate( Vec3d( 0, 0, 1), 45 );
    //table->rotate( Vec3d( 1,0,0),270);
    table->rotate( Vec3d( 0,1,0),0);
    table->scaleTo( 75 );
    table->move( Vec3d( 40,40,500) );
    table->setMinPoint({ 0, -50, 0 }, 1);
    table->setMaxPoint({ 0, 0, 600 }, 2);
    table->setMaxPoint({ 80, 0, 0 }, 0);
    table->setMaterial( { GRAY, 1 , 0 } );
    meshes.push_back( table );

    // -80, 80 ; -50, 50; 0, 600
    ////Mirror
    Vec3d moveVec = { 35,-5,0};
    RGB colorRam = GRAY;
    auto* mirrorBottom = new CubeMesh( Vec3d(0, 0, 595), Vec3d(30, 2, 600) );
    mirrorBottom->setMaterial( { colorRam, 1 , 0 } );
    mirrorBottom->move(moveVec);
    meshes.push_back( mirrorBottom );

    auto* mirrorLeft = new CubeMesh( Vec3d(0, 2, 595), Vec3d(2, 37, 600) );
    mirrorLeft->setMaterial( { colorRam, 1 , 0 } );
    mirrorLeft->move(moveVec);
    meshes.push_back( mirrorLeft );

    auto* mirrorRight = new CubeMesh( Vec3d(28, 2, 595), Vec3d(30, 37, 600) );
    mirrorRight->setMaterial( { colorRam, 1 , 0 } );
    mirrorRight->move(moveVec);
    meshes.push_back( mirrorRight );

    auto* mirrorTop = new CubeMesh( Vec3d(2, 35, 595), Vec3d(28, 37, 600) );
    mirrorTop->setMaterial( { colorRam, 1 , 0 } );
    mirrorTop->move(moveVec);
    meshes.push_back( mirrorTop );

    auto* mirror = new CubeMesh( Vec3d(2, 2, 598), Vec3d(30, 35, 600) );
    mirror->setMaterial( { GRAY, 1 , 1 } );
    mirror->move(moveVec);
    meshes.push_back( mirror );

    //CUBE
    auto* cube = new CubeMesh( Vec3d(2, 2, 592), Vec3d(12, 4, 602) );
    cube->setMaterial( { GRAY, 1 , 0 } );
    cube->move({14,-14,-20});
    meshes.push_back( cube );

    auto* plane = new Mesh();
    plane->loadMesh( "/home/auser/dev/src/Collection/Models/plane/model.obj" );
    //plane->rotate( Vec3d( 0, 0, 1), 10 );
    plane->rotate( Vec3d( 1,0,0),-15);
    plane->rotate( Vec3d( 0,1,0),-140);
    plane->move( Vec3d( 20,0,576) );
    plane->setMinPoint( Vec3d( 0,-63,0), 1 );
    plane->scaleTo( 15 );
    plane->setMaterial( { GRAY, 1 , 0 } );
    meshes.push_back( plane );

////LIGHTS

    //lights.push_back( new PointLight( Vec3d(0,25,450), 0.55));
    int lightWidth = 20;
    lights.push_back( new SpotLight( Vec3d(0 - lightWidth,45,480 - lightWidth), Vec3d(0 + lightWidth,45,480 + lightWidth), 0.70));

//    lights.push_back( new PointLight( Vec3d(-75,35,595), 0.15));
//    lights.push_back( new PointLight( Vec3d(75,35,595), 0.15));
//    lights.push_back( new PointLight( Vec3d(-75,35,5), 0.15));
//    lights.push_back( new PointLight( Vec3d(75,35,5), 0.15));
////LOADING...
    loadScene( scene, meshes, lights );
    rayTracer = new RayTracer( cam, scene, canvas, d, numAS, numLS );
}


void ratScene( RayTracer*& rayTracer, int w, int h, int d, int numAS, int numLS ) {
    Camera* cam = new Camera( Vec3d(0,0,0 ), Vec3d(0,0,1), 6000,3200,2000 );
    Scene* scene = new Scene();
    Canvas* canvas = new Canvas( w, h );
    Vector<Mesh*> meshes;
    Vector<Light*> lights;

    auto* rat = new Mesh();
    rat->loadMesh( "/home/auser/dev/src/Collection/Models/rat/model.obj" );
    //rat->rotate( Vec3d( 0, 0, 1), 45 );
    rat->rotate( Vec3d( 1,0,0),270);
    rat->rotate( Vec3d( 0,1,0),145);
    rat->move( Vec3d( 0,0,500) );
    rat->setMaterial( { RGB( 130, 130, 130 ), 1 , 0 } );
    meshes.push_back( rat );

    lights.push_back( new PointLight( Vec3d(20 ,0,0), 0.5));
    loadScene( scene, meshes, lights );
    rayTracer = new RayTracer( cam, scene, canvas, d, numAS, numLS );
}


void tableScene( RayTracer*& rayTracer, int w, int h, int d, int numAS, int numLS ) {
    Camera* cam = new Camera( Vec3d(0,0,0 ), Vec3d(0,0,1), 6000,3200,2000 );
    Scene* scene = new Scene();
    Canvas* canvas = new Canvas( w, h );
    Vector<Mesh*> meshes;
    Vector<Light*> lights;

    auto* table = new Mesh();
    table->loadMesh( "/home/auser/dev/src/Collection/Models/table/model.obj" );
    //table->rotate( Vec3d( 0, 0, 1), 45 );
    //table->rotate( Vec3d( 1,0,0),270);
    table->rotate( Vec3d( 0,1,0),-120);
    table->move( Vec3d( 40,40,1000) );
    table->setMaterial( { RGB( 130, 130, 130 ), 1 , 0 } );
    meshes.push_back( table );

    lights.push_back( new PointLight( Vec3d(20 ,0,0), 0.5));
    loadScene( scene, meshes, lights );
    rayTracer = new RayTracer( cam, scene, canvas, d, numAS, numLS );
}

void bookScene( RayTracer*& rayTracer, int w, int h, int d, int numAS, int numLS ) {
    Camera* cam = new Camera( Vec3d(0,0,0 ), Vec3d(0,0,1), 6000,3200,2000 );
    Scene* scene = new Scene();
    Canvas* canvas = new Canvas( w, h );
    Vector<Mesh*> meshes;
    Vector<Light*> lights;

    auto* book = new Mesh();
    book->loadMesh( "/home/auser/dev/src/Collection/Models/book/model.obj" );
    //book->rotate( Vec3d( 0, 0, 1), 45 );
    book->rotate( Vec3d( 1,0,0),-30);
    book->rotate( Vec3d( 0,1,0),-130);
    book->move( Vec3d( 40,0,1000) );
    book->setMaterial( { RGB( 130, 130, 130 ), 1 , 0 } );
    meshes.push_back( book );

    lights.push_back( new PointLight( Vec3d(20 ,0,0), 0.5));
    loadScene( scene, meshes, lights );
    rayTracer = new RayTracer( cam, scene, canvas, d, numAS, numLS );
}

void sandwichScene( RayTracer*& rayTracer, int w, int h, int d, int numAS, int numLS ) {
    Camera* cam = new Camera( Vec3d(0,0,0 ), Vec3d(0,0,1), 6000,3200,2000 );
    Scene* scene = new Scene();
    Canvas* canvas = new Canvas( w, h );
    Vector<Mesh*> meshes;
    Vector<Light*> lights;

    auto* sandwich = new Mesh();
    sandwich->loadMesh( "/home/auser/dev/src/Collection/Models/sandwich/model.obj" );
    sandwich->rotate( Vec3d( 0, 0, 1), 180 );
    sandwich->rotate( Vec3d( 1,0,0),-90);
    sandwich->rotate( Vec3d( 0,1,0),-70);
    sandwich->move( Vec3d( 40,0,1000) );
    sandwich->setMaterial( { RGB( 130, 130, 130 ), 1 , 0 } );
    meshes.push_back( sandwich );

    lights.push_back( new PointLight( Vec3d(20 ,0,0), 0.5));
    loadScene( scene, meshes, lights );
    rayTracer = new RayTracer( cam, scene, canvas, d, numAS, numLS );
}

void cartScene( RayTracer*& rayTracer, int w, int h, int d, int numAS, int numLS ) {
    Camera* cam = new Camera( Vec3d(0,0,0 ), Vec3d(0,0,1), 6000,3200,2000 );
    Scene* scene = new Scene();
    Canvas* canvas = new Canvas( w, h );
    Vector<Mesh*> meshes;
    Vector<Light*> lights;

    auto* cart = new Mesh();
    cart->loadMesh( "/home/auser/dev/src/Collection/Models/telega/model.obj" );
    //cart->rotate( Vec3d( 0, 0, 1), 45 );
    cart->rotate( Vec3d( 1,0,0),-90);
    cart->rotate( Vec3d( 0,1,0),60);
    cart->move( Vec3d( 40,0,1000) );
    cart->setMaterial( { RGB( 130, 130, 130 ), 1 , 0 } );
    meshes.push_back( cart );

    lights.push_back( new PointLight( Vec3d(20 ,0,0), 0.5));
    loadScene( scene, meshes, lights );
    rayTracer = new RayTracer( cam, scene, canvas, d, numAS, numLS );
}

void sksScene( RayTracer*& rayTracer, int w, int h, int d, int numAS, int numLS ) {
    Camera* cam = new Camera( Vec3d(0,0,0 ), Vec3d(0,0,1), 6000,3200,2000 );
    Scene* scene = new Scene();
    Canvas* canvas = new Canvas( w, h );
    Vector<Mesh*> meshes;
    Vector<Light*> lights;

    auto* sks = new Mesh();
    sks->loadMesh( "/home/auser/dev/src/Collection/Models/sks/model.obj" );
    //sks->rotate( Vec3d( 0, 0, 1), 45 );
    sks->rotate( Vec3d( 1,0,0),-90);
    sks->rotate( Vec3d( 0,1,0),-20);
    sks->move( Vec3d( 0,0,600) );
    sks->setMaterial( { RGB( 130, 130, 130 ), 1 , 0 } );
    meshes.push_back( sks );

    lights.push_back( new PointLight( Vec3d(20 ,0,0), 0.5));
    loadScene( scene, meshes, lights );
    rayTracer = new RayTracer( cam, scene, canvas, d, numAS, numLS );
}

void dogScene( RayTracer*& rayTracer, int w, int h, int d, int numAS, int numLS ) {
    Camera* cam = new Camera( Vec3d(0,0,0 ), Vec3d(0,0,1), 6000,3200,2000 );
    Scene* scene = new Scene();
    Canvas* canvas = new Canvas( w, h );
    Vector<Mesh*> meshes;
    Vector<Light*> lights;

    auto* dog = new Mesh();
    dog->loadMesh( "/home/auser/dev/src/Collection/Models/dog/model.obj" );
    //dog->rotate( Vec3d( 0, 0, 1), 45 );
    dog->rotate( Vec3d( 1,0,0),-90);
    dog->rotate( Vec3d( 0,1,0),-235);
    dog->move( Vec3d( 0,0,700) );
    dog->setMaterial( { RGB( 130, 130, 130 ), 1 , 0 } );
    meshes.push_back( dog );

    lights.push_back( new PointLight( Vec3d(20 ,0,0), 0.5));
    loadScene( scene, meshes, lights );
    rayTracer = new RayTracer( cam, scene, canvas, d, numAS, numLS );
}

void planeScene( RayTracer*& rayTracer, int w, int h, int d, int numAS, int numLS ) {
    Camera* cam = new Camera( Vec3d(0,0,0 ), Vec3d(0,0,1), 6000,3200,2000 );
    Scene* scene = new Scene();
    Canvas* canvas = new Canvas( w, h );
    Vector<Mesh*> meshes;
    Vector<Light*> lights;

    auto* plane = new Mesh();
    plane->loadMesh( "/home/auser/dev/src/Collection/Models/plane/model.obj" );
    //dog->rotate( Vec3d( 0, 0, 1), 45 );
    plane->rotate( Vec3d( 1,0,0),10);
    plane->rotate( Vec3d( 0,1,0),-200);
    plane->move( Vec3d( 0,0,600) );
    plane->setMaterial( { RGB( 130, 130, 130 ), 1 , 0 } );
    meshes.push_back( plane );
    lights.push_back( new PointLight( Vec3d(20 ,0,0), 0.5));
    loadScene( scene, meshes, lights );
    rayTracer = new RayTracer( cam, scene, canvas, d, numAS, numLS );
}

void modelRoomScene( RayTracer*& rayTracer, int w, int h, int d, int numAS, int numLS ) {
    Camera* cam = new Camera( Vec3d(0,0,0 ), Vec3d(0,0,1), 6000,3200,2000 );
    Scene* scene = new Scene();
    Canvas* canvas = new Canvas( w, h );
    Vector<Mesh*> meshes;
    Vector<Light*> lights;

    auto* room = new Mesh();
    room->loadMesh( "/home/auser/dev/src/Collection/Models/cottage/Cottage_FREE.obj" );
    //room->rotate( Vec3d( 0, 0, 1), 45 );
    //room->rotate( Vec3d( 1,0,0),10);
    //room->rotate( Vec3d( 0,1,0),-200);
    room->move( Vec3d( 0,0,0) );
    room->setMaterial( { RGB( 130, 130, 130 ), 1 , 0 } );
    meshes.push_back( room );
    lights.push_back( new PointLight( Vec3d(20 ,0,0), 0.5));
    loadScene( scene, meshes, lights );
    rayTracer = new RayTracer( cam, scene, canvas, d, numAS, numLS );
}
void cottageScene( RayTracer*& rayTracer, int w, int h, int d, int numAS, int numLS ) {
    Camera* cam = new Camera( Vec3d(0,0,0 ), Vec3d(0,0,1), 6000,3200,2000 );
    Scene* scene = new Scene();
    Canvas* canvas = new Canvas( w, h );
    Vector<Mesh*> meshes;
    Vector<Light*> lights;

    auto* cottage = new Mesh();
    cottage->loadMesh( "/home/auser/dev/src/Collection/Models/underground/underground.obj" );
    //cottage->rotate( Vec3d( 0, 0, 1), 45 );
    //cottage->rotate( Vec3d( 1,0,0),10);
    cottage->rotate( Vec3d( 0,1,0),-260);
    cottage->move( Vec3d( 0,0,800) );
    cottage->setMaterial( { RGB( 130, 130, 130 ), 1 , 0 } );
    meshes.push_back( cottage );
    lights.push_back( new PointLight( Vec3d(20 ,0,0), 0.5));
    loadScene( scene, meshes, lights );
    rayTracer = new RayTracer( cam, scene, canvas, d, numAS, numLS );
}

void carScene( RayTracer*& rayTracer, int w, int h, int d, int numAS, int numLS ) {
    Camera* cam = new Camera( Vec3d(0,0,0 ), Vec3d(0,0,1), 6000,3200,2000 );
    Scene* scene = new Scene();
    Canvas* canvas = new Canvas( w, h );
    Vector<Mesh*> meshes;
    Vector<Light*> lights;

    auto* car = new Mesh();
    car->loadMesh( "/home/auser/dev/src/Collection/Models/koenigsegg/model.obj" );
    //car->rotate( Vec3d( 0, 0, 1), 45 );
    //car->rotate( Vec3d( 1,0,0),10);
    car->rotate( Vec3d( 0,1,0),-210);
    car->move( Vec3d( 0,0,600) );
    car->setMaterial( { RGB( 130, 130, 130 ), 1 , 0 } );
    meshes.push_back( car );
    lights.push_back( new PointLight( Vec3d(20 ,0,0), 0.5));
    loadScene( scene, meshes, lights );
    rayTracer = new RayTracer( cam, scene, canvas, d, numAS, numLS );
}

void hardScene( RayTracer*& rayTracer, int w, int h, int d, int numAS, int numLS ) {
    Camera* cam = new Camera( Vec3d(0,0,0 ), Vec3d(0,0,1), 6000,3200,2000 );
    Scene* scene = new Scene();
    Canvas* canvas = new Canvas( w, h );
    Vector<Mesh*> meshes;
    Vector<Light*> lights;
    auto* table = new Mesh();
    table->loadMesh( "/home/auser/dev/src/Collection/Models/table/model.obj" );
    //table->rotate( Vec3d( 0, 0, 1), 20 );
    //table->rotate( Vec3d( 1,0,0),20);
    table->rotate( Vec3d( 0,1,0),-120);
    table->move( Vec3d( 40,40,1200) );
    table->setMaterial( { RGB( 130, 130, 130 ), 1 , 0 } );



    auto* sks = new Mesh();
    sks->loadMesh( "/home/auser/dev/src/Collection/Models/sks/model.obj" );
    sks->rotate( Vec3d( 0, 0, 1), 0 );
    sks->rotate( Vec3d( 1,0,0),0);
    sks->rotate( Vec3d( 0,1,0),0);
    sks->move( Vec3d( -25,19,1080) );
    sks->setMaterial( { RGB( 130, 130, 130 ), 1 , 0 } );

    meshes.push_back( table );
    meshes.push_back( sks );


    lights.push_back( new PointLight( Vec3d(20 ,0,0), 0.5));
    loadScene( scene, meshes, lights );
    rayTracer = new RayTracer( cam, scene, canvas, d, numAS, numLS );
}

void audiScene( RayTracer*& rayTracer, int w, int h, int d, int numAS, int numLS ) {
    double FOV = 67.38;
    double dV = w / 2 / tan( FOV * M_PI / 180 / 2 );
    Camera* cam = new Camera( Vec3d(0,10,0 ), Vec3d(0,0,1), dV,w,h );
    //Camera* cam = new Camera( Vec3d(0,0,0 ), Vec3d(0,0,1), 6000,3200,2000 );
    Scene* scene = new Scene();
    Canvas* canvas = new Canvas( w, h );

    Vector<Mesh*> meshes;
    Vector<Light*> lights;
    double roomRefl = 0;
    double roomHeight = 150;
    double roomWidth = 300;
    double roomLength = 600;
    double du = 10;
    double du2 = du / 2;

    double roomHeight2 = roomHeight / 2;
    double roomWidth2 = roomWidth / 2;
    double roomLength2 = roomLength / 2;


    Material whiteBrick;
    whiteBrick.setTexture( "/home/auser/dev/src/Collection/Textures/WhiteBrick/" );
    Material whiteFloor;
    whiteFloor.setTexture( "/home/auser/dev/src/Collection/Textures/StoneWhiteFloor/" );
////right
    meshes.push_back( new CubeMesh( Vec3d(roomWidth2, -roomHeight2, 0), Vec3d(roomWidth2 + du, roomHeight2, roomLength),
                                    whiteBrick ) );
////left
    meshes.push_back(new CubeMesh( Vec3d(-roomWidth2 - du, -roomHeight2, 0), Vec3d(-roomWidth2, roomHeight2, roomLength),
                                   whiteBrick ) );
////front
    meshes.push_back(new CubeMesh( Vec3d(-roomWidth2, -roomHeight2, roomLength), Vec3d(roomWidth2, roomHeight2, roomLength + du),
                                   whiteBrick ) );
////back
    meshes.push_back(new CubeMesh( Vec3d(-roomWidth2, -roomHeight2, -du), Vec3d(roomWidth2, roomHeight2, 0),
                                   whiteBrick ) );
////down
    meshes.push_back(new CubeMesh( Vec3d(-roomWidth2, -roomHeight2 - du, 0), Vec3d(roomWidth2, -roomHeight2, roomLength),
                                   whiteFloor ) );
////up
    meshes.push_back(new CubeMesh( Vec3d(-roomWidth2, roomHeight2, 0), Vec3d(roomWidth2, roomHeight2 + du, roomLength),
                                   { GRAY, -1 , roomRefl } ) );

////AUDI
    auto* audi = new GroupOfMeshes();
    audi->loadMesh( "/home/auser/dev/src/Collection/Models/audi/audi.obj" );
    audi->scaleTo( 250 );
    audi->rotate( Vec3d( 0,1,0), 145);
    audi->moveTo(Vec3d( 0,0, roomLength / 2 + 40  ) );
    audi->setMinPoint( Vec3d( 0,-roomHeight2,0), 1 );
    audi->setMaterial( { GRAY, -1 , 1 } );
    auto bbox = audi->getBBox();
    std::cout << " bbox - " << bbox.pMin << " " << bbox.pMax << std::endl;
    // mb 75 is light
    std::vector< std::pair< std::vector<int>, Material > > drawing;
    Material steel;
    steel.setTexture( "/home/auser/dev/src/Collection/Textures/Steel/" );
    Material blackMetal;
    blackMetal.setTexture( "/home/auser/dev/src/Collection/Textures/BlackMetal/" );
    Material damagedGold;
    damagedGold.setTexture( "/home/auser/dev/src/Collection/Textures/DamagedGold/" );
    Material lightBlueMetal;
    lightBlueMetal.setTexture( "/home/auser/dev/src/Collection/Textures/LightBlueMetal/" );
    Material blueMetal;
    blueMetal.setTexture( "/home/auser/dev/src/Collection/Textures/BlueMetal/" );
    Material pinkMetal;
    pinkMetal.setTexture( "/home/auser/dev/src/Collection/Textures/PinkMetal/" );


    std::vector<int> bodyIndexes =
            { 0, 1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 18, 21, 24, 28, 29, 30, 54, 55, 56, 69, 70, 71, 75, 83, 84, 88, 89, 106, 107, 110, 111, 114, 127, 168 };
    Material body = lightBlueMetal;
    drawing.push_back( { bodyIndexes, body } );

    std::vector<int> tireIndexes = { 74, 103 };
    Material tire = { BLACK, -1, 1 };
    drawing.push_back( { tireIndexes, tire } );

    std::vector<int> diskIndexes = { 76, 77, 104, 105 };
    Material disk = steel;
    drawing.push_back( { diskIndexes, disk } );

    std::vector<int> boltIndexes = { 61, 62, 94, 95, 96 };
    Material bolt = damagedGold;
    drawing.push_back( { boltIndexes, bolt } );
    //BREAKS

    //
    std::vector<int> backGridIndexes = { 9, 11, 12, 27, 85, 86 };
    Material backGrid = { BLACK, -1, 1 };
    drawing.push_back( { backGridIndexes, backGrid } );

    std::vector<int> gridIndexes = { 10, 13, 23, 26, 57, 113 };
    Material grid = blackMetal;
    drawing.push_back( { gridIndexes, grid } );

    std::vector<int> handlerIndexes = { 115, 116 };
    Material handler = body;
    drawing.push_back( { handlerIndexes, handler } );

    std::vector<int> lightIndexes = { 92, 137 };
    Material light = { WHITE, -1, 1 };
    light.setMetalness( 1 );
    drawing.push_back( { lightIndexes, light } );

    std::vector<int> mirrorIndexes = { 112 };
    Material mirror = { WHITE, -1, 0.001 };
    mirror.setMetalness( 1 );
    drawing.push_back( { mirrorIndexes, mirror } );

    std::vector<int> windowIndexes = { 17, 19, 22 };
    Material window = { BLACK, -1, 0.01 };
    window.setMetalness( 1 );
    drawing.push_back( { windowIndexes, window } );

    std::vector<int> edgingIndexes = { 16, 20, 73, 82, 90, 91 };
    Material edging = { BLACK, -1, 1 };
    drawing.push_back( { edgingIndexes, edging } );

    std::vector<int> signIndexes = { 58, 87, 93, 117, 118, 120, 121, 122, 123, 125 };
    Material sign = steel;
    drawing.push_back( { signIndexes, sign } );

    std::vector<int> exhaustIndexes = { 25 };
    Material exhaust = steel;
    drawing.push_back( { exhaustIndexes, exhaust } );

    for ( const auto& [ vecInd, mat ]: drawing ) {
        for ( auto ind: vecInd ) {
            audi->setMaterial( mat, ind );
        }
    }

    for ( auto mesh: audi->getMeshes() ) {
        meshes.push_back( mesh );
    }

////LIGHTS

    double lightWidth = 80;
    double lightLenght = 20;
    double par = roomLength / 6;
    double intensity = 2;
    meshes.push_back( new CubeMesh( Vec3d( -lightWidth, roomHeight2 - du2, par - lightLenght),
                                    Vec3d( lightWidth, roomHeight2 - du2, par + lightLenght),
                                    { WHITE, intensity }));

    meshes.push_back( new CubeMesh( Vec3d( -lightWidth, roomHeight2 - du2, 3 * par - lightLenght),
                                    Vec3d( lightWidth, roomHeight2 - du2, 3 * par + lightLenght),
                                    { WHITE, intensity }));

    meshes.push_back( new CubeMesh( Vec3d( -lightWidth, roomHeight2 - du2, 5 * par - lightLenght),
                                    Vec3d( lightWidth, roomHeight2 - du2, 5 * par + lightLenght),
                                    { WHITE, intensity }));
////LOADING...
    loadScene( scene, meshes, lights );
    rayTracer = new RayTracer( cam, scene, canvas, d, numAS, numLS );
}

void sphereRoomScene( RayTracer*& rayTracer, int w, int h, int d, int numAS, int numLS ) {
    double FOV = 67.38;
    double dV = w / 2 / tan( FOV * M_PI / 180 / 2 );
    Camera* cam = new Camera( Vec3d(0,10,0 ), Vec3d(0,0,1), dV,w,h );
    //Camera* cam = new Camera( Vec3d(0,0,0 ), Vec3d(0,0,1), 6000,3200,2000 );
    Scene* scene = new Scene();
    Canvas* canvas = new Canvas( w, h );

    Vector<Mesh*> meshes;
    Vector<Light*> lights;
    double roomRefl = 1;

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
    Material damagedGold = {GRAY, -1, 0 };
    damagedGold.setTexture( "/home/auser/dev/src/Collection/Textures/DamagedGold/");

////right
    meshes.push_back( new CubeMesh( Vec3d(70, -50, 0), Vec3d(80, 70, 600),
                                   wall ) );
////left
    meshes.push_back(new CubeMesh( Vec3d(-80, -50, 0), Vec3d(-70, 70, 600),
                                   wall ) );
////front
    meshes.push_back(new CubeMesh( Vec3d(-100, -50, 290), Vec3d(100, 70, 300),
                                   wall ) );
////back
    meshes.push_back(new CubeMesh( Vec3d(-100, -50, -10), Vec3d(100, 70, 0),
                                   wall ) );
////floor
    meshes.push_back(new CubeMesh( Vec3d(-100, -70, 0), Vec3d(100, -50, 620),
                                   floor ) );
////ceil
    meshes.push_back(new CubeMesh( Vec3d(-100, 70, 0), Vec3d(100, 90, 620),
                                   wall ) );


    ////RAND BLOCK
    auto* randBlockForward = new CubeMesh( Vec3d(-15, -50, 310), Vec3d(15, -30, 340) );
    randBlockForward->moveTo( Vec3d(20, -40, 175) );
    randBlockForward->scaleTo( Vec3d(30,100,30) );
    randBlockForward->rotate( Vec3d( 0,1,0), 25);
    randBlockForward->setMaterial( carpet );
    meshes.push_back(randBlockForward );

    auto* randBlockForward2 = new CubeMesh( Vec3d(-15, -50, 310), Vec3d(15, -30, 340) );
    randBlockForward2->moveTo( Vec3d(-35, -40, 205) );
    randBlockForward2->scaleTo( Vec3d(30,260,30) );
    randBlockForward2->rotate( Vec3d( 0,1,0), 45);
    randBlockForward2->setMaterial( marble );
    meshes.push_back(randBlockForward2 );


////Spheres
//    Vector<Sphere* > spheres;
//    spheres.push_back( new Sphere( 20, Vec3d(20, 0, 175), damagedGold ) );


////LIGHTS

//    lights.push_back( new PointLight( Vec3d(0,65,150), 0.55));
    int lightWidth = 20;
    meshes.push_back( new CubeMesh( Vec3d(0 - lightWidth,64,150 - lightWidth), Vec3d(0 + lightWidth,65,150 + lightWidth), { WHITE, 1.6 }));

////LOADING...
//    for ( auto sphere: spheres ) {
//        scene->add( sphere );
//    }
    loadScene( scene, meshes, lights );
    rayTracer = new RayTracer( cam, scene, canvas, d, numAS, numLS );
}


void dragonScene( RayTracer*& rayTracer, int w, int h, int d, int numAS, int numLS ) {
    double FOV = 67.38;
    double dV = w / 2 / tan( FOV * M_PI / 180 / 2 );
    Camera* cam = new Camera( Vec3d(0,10,0 ), Vec3d(0,0,1), dV,w,h );
    //Camera* cam = new Camera( Vec3d(0,0,0 ), Vec3d(0,0,1), 6000,3200,2000 );
    Scene* scene = new Scene();
    Canvas* canvas = new Canvas( w, h );

    Vector<Mesh*> meshes;
    Vector<Light*> lights;
    double roomRefl = 0;
////right
    meshes.push_back( new CubeMesh( Vec3d(70, -50, 0), Vec3d(80, 70, 600),
                                    { GREEN, -1 , roomRefl } ) );
////left
    meshes.push_back(new CubeMesh( Vec3d(-80, -50, 0), Vec3d(-70, 70, 600),
                                   { RED, -1 , roomRefl } ) );
////front
    meshes.push_back(new CubeMesh( Vec3d(-100, -50, 290), Vec3d(100, 70, 300),
                                   { GRAY, -1, roomRefl } ) );
////back
    meshes.push_back(new CubeMesh( Vec3d(-100, -50, -10), Vec3d(100, 70, 0),
                                   { GRAY, -1 , roomRefl } ) );
////down
    meshes.push_back(new CubeMesh( Vec3d(-100, -70, 0), Vec3d(100, -50, 620),
                                   { GRAY, -1 , roomRefl } ) );
////up
    meshes.push_back(new CubeMesh( Vec3d(-100, 70, 0), Vec3d(100, 90, 620),
                                   { GRAY, -1 , roomRefl } ) );

////RAND BLOCK
    auto* dragon = new Mesh();
    dragon->loadMesh( "/home/auser/dev/src/Collection/Models/dragon/armadillo.obj" );
    dragon->setMaterial(  { GRAY, -1 , roomRefl } );
    dragon->scaleTo(100 );
    dragon->moveTo( {0,0,150} );
    dragon->rotate( Vec3d( 1,0,0),-30);
    dragon->setMinPoint( {0,-50,0}, 1 );
    meshes.push_back( dragon );
////LIGHTS

    lights.push_back( new PointLight( Vec3d(0,65,150), 0.6));
    int lightWidth = 20;
    //lights.push_back( new SpotLight( Vec3d(0 - lightWidth,65,180 - lightWidth), Vec3d(0 + lightWidth,65,180 + lightWidth), 0.7));

////LOADING...
    loadScene( scene, meshes, lights );
    rayTracer = new RayTracer( cam, scene, canvas, d, numAS, numLS );
}


void testScene( RayTracer*& rayTracer, int w, int h, int d, int numAS, int numLS ) {
    double FOV = 67.38;
    double dV = w / 2 / tan( FOV * M_PI / 180 / 2 );
    Camera* cam = new Camera( Vec3d(0,10,0 ), Vec3d(0,0,1), dV,w,h );
    //Camera* cam = new Camera( Vec3d(0,0,0 ), Vec3d(0,0,1), 6000,3200,2000 );
    Scene* scene = new Scene();
    Canvas* canvas = new Canvas( w, h );

    Vector<Mesh*> meshes;
    Vector<Light*> lights;
    double roomRefl = 1;

    Material floor = {GRAY, -1, 0 };
    floor.setTexture( "/home/auser/dev/src/Collection/Textures/WoodFloorBright/");

////right
    meshes.push_back( new CubeMesh( Vec3d(70, -50, 0), Vec3d(80, 70, 600),
                                    floor ) );
////left
    meshes.push_back(new CubeMesh( Vec3d(-80, -50, 0), Vec3d(-70, 70, 600),
                                   floor ) );
////front
    meshes.push_back(new CubeMesh( Vec3d(-100, -50, 290), Vec3d(100, 70, 300),
                                   floor ) );
////back
    meshes.push_back(new CubeMesh( Vec3d(-100, -50, -10), Vec3d(100, 70, 0),
                                   floor ) );
////floor
    meshes.push_back(new CubeMesh( Vec3d(-100, -70, 0), Vec3d(100, -50, 620),
                                   floor ) );
////ceil
    meshes.push_back(new CubeMesh( Vec3d(-100, 70, 0), Vec3d(100, 90, 620),
                                   floor ) );


    ////RAND BLOCK
    auto* m = new Mesh();
    m->loadMesh( "/home/auser/dev/src/Collection/Models/torus.obj" );
    m->setMaterial( { GRAY, -1, 1 } );
    m->scaleTo( 50 );
    m->moveTo( { 0, 0, 150 } );
    meshes.push_back( m );

////LIGHTS

    lights.push_back( new PointLight( Vec3d(0,0,0), 3));
    lights.push_back( new PointLight( Vec3d(0,0,290), 3));
    lights.push_back( new PointLight( Vec3d(0,65,250), 3));
//    int lightWidth = 20;
//    meshes.push_back( new CubeMesh( Vec3d(0 - lightWidth,64,150 - lightWidth), Vec3d(0 + lightWidth,65,150 + lightWidth), { WHITE, 1.2 }));

    loadScene( scene, meshes, lights );
    rayTracer = new RayTracer( cam, scene, canvas, d, numAS, numLS );
}

int main( int argc, char* argv[] ) {
    setenv("OMP_PROC_BIND", "spread", 1);
    setenv("OMP_PLACES", "threads", 1);
    Kokkos::initialize(argc, argv); {
    std::cout << "Default Execution Space: " << Kokkos::DefaultExecutionSpace::name() << std::endl;
    srand(time( nullptr ));
//    srand(0);
    RayTracer* rayTracer = nullptr;
    ////OPTIONS

    ////RESOLUTION
    //int w = 8 ; int h = 5;
    int w = 240 ; int h = 150;
    //int w = 640 ; int h = 400; //53 sec //
    //int w = 960 ; int h = 600; // 42 sec
    //int w = 1920 ; int h = 1200;
    //int w = 3200; int h = 2000;
////49 sec // 46 sec
    
    // 22 sec /
    // ( 960x600 (2,5,2) audi scene) - 42 sec / 47 sec /

    ////NUM SAMPLES
    int depth = 2;
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
    audiScene( rayTracer, w, h, depth, ambientSamples, lightSamples ); //720 sec// 4 sec
    //sphereRoomScene( rayTracer, w, h, depth, ambientSamples, lightSamples );//57 sec // 13.6 sec
    //dragonScene( rayTracer, w, h, depth, ambientSamples, lightSamples );//57 sec // 13.6 sec
    //testScene( rayTracer, w, h, depth, ambientSamples, lightSamples );//57 sec // 13.6 sec
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> loadTime = end - start;
    std::cout << "Model loads "<< loadTime.count() << " seconds" << std::endl;
    start = std::chrono::high_resolution_clock::now();;
    rayTracer->render( RayTracer::SERIAL );
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
// net room scene
// 2 5 2 parameters
// 3200x2000
//time - 168 sec
// 1920x1200
//time - 62.7


//audi scene
// 2 5 2 parametersz
//time - 502.625