//
// Created by auser on 8/4/24.
//
#include <iostream>
#include "RayTracer.h"
#include "cstdlib"
#include "Denoiser.h"
#include "LuaLoader.h"

int main( int argc, char* argv[] ) {
    setenv("OMP_PROC_BIND", "spread", 1 );
    setenv("OMP_PLACES", "threads", 1 );
    lua_State* L = luaL_newstate();
    luaL_openlibs(L);

    if (luaL_dofile(L, "/home/auser/dev/src/Collection/Applications/Graphics/netRoom.lua") != LUA_OK) {
        std::cerr << lua_tostring(L, -1) << std::endl;
        lua_close(L);
        return 1;
    }
    Scene* scene = new Scene();
    loadScene( L, scene );
    Canvas* canvas = loadCanvas( L );
    Camera* camera = loadCamera( L );
    auto settings = loadSettings( L );
   // scene->add( new PointLight( Vector3f(-3500,0,0 ), 9999999 ) );
    Kokkos::initialize(argc, argv); {
    RayTracer* rayTracer = new RayTracer( camera, scene, canvas, settings[0], settings[1], settings[2] );
    rayTracer->render( RayTracer::PARALLEL );
    rayTracer->getCanvas()->saveToPNG( "out.png" );
        if ( settings[3] == 1 ) {
            Denoiser::denoise( rayTracer->getCanvas()->getData(), rayTracer->getCanvas()->getW(), rayTracer->getCanvas()->getH() );
            rayTracer->getCanvas()->saveToPNG( "outDenoised.png" );
        }
    } Kokkos::finalize();

//    RayTracer rt;
//    loadCameraFromLua(L, rt);
//    loadLightsFromLua(L, rt);
//
//    rt.render();

    lua_close(L);
    return 0;
}