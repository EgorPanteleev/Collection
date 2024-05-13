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

void sphereScene( RayTracer*& rayTracer, int w, int h ) {
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



    //lights.push_back( new Light( Vector3f(-3500,0,0 ), 0.004 ));
    lights.push_back( new Light( Vector3f(-1000,0,0 ), 0.004 ));
    loadScene( scene, meshes, lights );
    rayTracer = new RayTracer( cam, scene, canvas );
}


void roomScene( RayTracer*& rayTracer, int w, int h ) {
    Camera* cam = new Camera( Vector3f(0,0,0 ), Vector3f(0,0,1), 6000,3200,2000 );
    Scene* scene = new Scene();
    Canvas* canvas = new Canvas( w, h );

    std::vector<BaseMesh*> meshes;
    std::vector<Light*> lights;
////right
    meshes.push_back( new CubeMesh( Vector3f(80, -50, 0), Vector3f(100, 50, 600),
                                    { GRAY, 1 , 0 } ) );
////left
    meshes.push_back(new CubeMesh( Vector3f(-100, -50, 0), Vector3f(-80, 50, 600),
                                   { GRAY, 1 , 0 } ) );
////back
    meshes.push_back(new CubeMesh( Vector3f(-100, -50, 600), Vector3f(100, 50, 600),
                                   { GRAY, 1 , 0 } ) );
////front
    meshes.push_back(new CubeMesh( Vector3f(-100, -50, 0), Vector3f(100, 50, 0),
                                   { GRAY, 1 , 0 } ) );
////down
    meshes.push_back(new CubeMesh( Vector3f(-100, -70, 0), Vector3f(100, -50, 620),
                                   { GRAY, 1 , 0 } ) );
////up
    meshes.push_back(new CubeMesh( Vector3f(-100, 50, 0), Vector3f(100, 70, 620),
                                   { GRAY, 1 , 0 } ) );

////RAND BLOCK
    auto* randBlockForward = new CubeMesh( Vector3f(-15, -50, 310), Vector3f(15, -30, 340) );
    randBlockForward->moveTo( Vector3f(0, -40, 325) );
    randBlockForward->scaleTo( Vector3f(20,90,20) );
    randBlockForward->rotate( Vector3f( 0,1,0), 45);
    randBlockForward->move( Vector3f(-10,0,0));
    randBlockForward->setMaterial({RED, 1 , 0});
    //randBlockForward->scaleTo( 200 );
    meshes.push_back(randBlockForward );
////LIGHTS
    lights.push_back( new Light( Vector3f(-75,35,595), 0.15));
    lights.push_back( new Light( Vector3f(75,35,595), 0.15));
    lights.push_back( new Light( Vector3f(-75,35,5), 0.15));
    lights.push_back( new Light( Vector3f(75,35,5), 0.15));
////LOADING...
    loadScene( scene, meshes, lights );
    rayTracer = new RayTracer( cam, scene, canvas );
}


void ratScene( RayTracer*& rayTracer, int w, int h ) {
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


    lights.push_back( new Light( Vector3f(20 ,0,0), 0.5));
    loadScene( scene, meshes, lights );
    rayTracer = new RayTracer( cam, scene, canvas );
}


void tableScene( RayTracer*& rayTracer, int w, int h ) {
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

    lights.push_back( new Light( Vector3f(20 ,0,0), 0.5));
    loadScene( scene, meshes, lights );
    rayTracer = new RayTracer( cam, scene, canvas );
}

void bookScene( RayTracer*& rayTracer, int w, int h ) {
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

    lights.push_back( new Light( Vector3f(20 ,0,0), 0.5));
    loadScene( scene, meshes, lights );
    rayTracer = new RayTracer( cam, scene, canvas );
}

void sandwichScene( RayTracer*& rayTracer, int w, int h ) {
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

    lights.push_back( new Light( Vector3f(20 ,0,0), 0.5));
    loadScene( scene, meshes, lights );
    rayTracer = new RayTracer( cam, scene, canvas );
}

void vagonScene( RayTracer*& rayTracer, int w, int h ) {
    Camera* cam = new Camera( Vector3f(0,0,0 ), Vector3f(0,0,1), 6000,3200,2000 );
    Scene* scene = new Scene();
    Canvas* canvas = new Canvas( w, h );
    std::vector<BaseMesh*> meshes;
    std::vector<Light*> lights;

    auto* vagon = new TriangularMesh();
    vagon->loadMesh( "/home/auser/dev/src/Collection/Models/telega/model.obj" );
    //vagon->rotate( Vector3f( 0, 0, 1), 45 );
    vagon->rotate( Vector3f( 1,0,0),-90);
    vagon->rotate( Vector3f( 0,1,0),60);
    vagon->move( Vector3f( 40,0,1000) );
    vagon->setMaterial( { RGB( 130, 130, 130 ), 1 , 0 } );
    meshes.push_back( vagon );

    lights.push_back( new Light( Vector3f(20 ,0,0), 0.5));
    loadScene( scene, meshes, lights );
    rayTracer = new RayTracer( cam, scene, canvas );
}

void sksScene( RayTracer*& rayTracer, int w, int h ) {
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

    lights.push_back( new Light( Vector3f(20 ,0,0), 0.5));
    loadScene( scene, meshes, lights );
    rayTracer = new RayTracer( cam, scene, canvas );
}

void dogScene( RayTracer*& rayTracer, int w, int h ) {
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

    lights.push_back( new Light( Vector3f(20 ,0,0), 0.5));
    loadScene( scene, meshes, lights );
    rayTracer = new RayTracer( cam, scene, canvas );
}

void planeScene( RayTracer*& rayTracer, int w, int h ) {
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
    plane->move( Vector3f( 0,-50,14800) );
    plane->setMaterial( { RGB( 130, 130, 130 ), 1 , 0 } );
    meshes.push_back( plane );
    int a = plane->getTriangles().size();
    lights.push_back( new Light( Vector3f(20 ,0,0), 0.5));
    loadScene( scene, meshes, lights );
    rayTracer = new RayTracer( cam, scene, canvas );
}

void hardScene( RayTracer*& rayTracer, int w, int h ) {
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

    lights.push_back( new Light( Vector3f(20 ,0,0), 0.5));
    loadScene( scene, meshes, lights );
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

//rat // table // book // sandwich // telega

int main() {
    RayTracer* rayTracer = nullptr;
    //int w = 8 ; int h = 5;
    //int w = 240 ; int h = 150;
    //int w = 640 ; int h = 400;
    //int w = 960 ; int h = 600;
    //int w = 1920 ; int h = 1200;
    int w = 3200 ; int h = 2000;
// room scene ( 960x600 ) - 18.1 / 15.5 / 9.7 / 9.3 / 7.3
// room scene ( 3200x2000 ) - idk / 95 /
// rat scene ( 3200x2000 ) - 100 / 79 / 4.6
    clock_t start = clock();
    //sphereScene( rayTracer, w, h );//
    //roomScene( rayTracer, w, h );//57 sec // 13.6 sec
    //ratScene( rayTracer, w, h );//2.3 sec // 1.7 sec
    //tableScene( rayTracer, w, h );//23 sec // 1.56 sec
    //bookScene( rayTracer, w, h );//130 sec // 31 sec
    //sandwichScene( rayTracer, w, h );//3.29 sec //2 sec
    //vagonScene( rayTracer, w, h );//118 sec // 1.96 sec
    sksScene( rayTracer, w, h );//182 sec //1.6 sec
    //dogScene( rayTracer, w, h );//10 sec //
    //planeScene( rayTracer, w, h );//357 sec
    //hardScene( rayTracer, w, h ); //720 sec
    clock_t end = clock();
    double seconds = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Model loads %f seconds\n", seconds);
    start = clock();
    //rayt.traceAllRaysWithThreads( 1);
    rayTracer->traceAllRays();
    end = clock();
    seconds = (double)(end - start) / CLOCKS_PER_SEC;
    printf("RayTracer works %f seconds\n", seconds);
    saveToBMP( rayTracer->getCanvas(), "out.bmp" );
   //delete
   //TODO mb need init rayTracer more
   //TODO think about camera, i think its bad right now
    return 0;
}