#include <iostream>
#include "RayTracer.h"
#include "CubeMesh.h"
#include "PointLight.h"
#include "SpotLight.h"
#include "cstdlib"
#include "Denoiser.h"
#include "GroupOfMeshes.h"
#include "Triangles.h"

void loadScene(Scene* scene, Vector <Mesh*>& meshes, Vector<Light*>& lights ) {
    for ( const auto mesh: meshes ) {
        scene->add( mesh );
    }
    for ( const auto light: lights ) {
        scene->add( light );
    }
}

void sphereScene( RayTracer*& rayTracer, int w, int h, const RayTracerParameters& params ) {
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
    rayTracer = new RayTracer( cam, scene, canvas, params );
}
void netRoomScene( RayTracer*& rayTracer, int w, int h, const RayTracerParameters& params ) {
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
    rayTracer = new RayTracer( cam, scene, canvas, params );
}

void audiScene( RayTracer*& rayTracer, int w, int h, const RayTracerParameters& params ) {
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
    rayTracer = new RayTracer( cam, scene, canvas, params );
}

void sphereRoomScene( RayTracer*& rayTracer, int w, int h, const RayTracerParameters& params ) {
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
    rayTracer = new RayTracer( cam, scene, canvas, params );
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
    //int w = 240 ; int h = 150;
    //int w = 640 ; int h = 400;
    //int w = 960 ; int h = 600;
    int w = 1920 ; int h = 1200;
    //int w = 3200; int h = 2000;


    ////NUM SAMPLES
    int depth = 2;
    int numSamples = 1;
    int ambientSamples = 5;
    int lightSamples = 2;

    RayTracerParameters params( depth, numSamples, ambientSamples, lightSamples );

    auto start = std::chrono::high_resolution_clock::now();
    //sphereScene( rayTracer, w, h, params );//
    netRoomScene( rayTracer, w, h, params );//57 sec // 13.6 sec
    //audiScene( rayTracer, w, h, params ); //720 sec// 4 sec
    //sphereRoomScene( rayTracer, w, h, params );//57 sec // 13.6 sec
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

// +-62 sec