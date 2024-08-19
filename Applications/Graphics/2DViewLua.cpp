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

    if (luaL_dofile(L, "/home/auser/dev/src/Collection/Applications/Graphics/spheres.lua") != LUA_OK) {
        std::cerr << lua_tostring(L, -1) << std::endl;
        lua_close(L);
        return 1;
    }
   // scene->add( new PointLight( Vector3f(-3500,0,0 ), 9999999 ) );
    Kokkos::initialize(argc, argv); {
        RayTracer* rayTracer = loadRayTracer( L );
        auto start = std::chrono::high_resolution_clock::now();;
        rayTracer->render( RayTracer::PARALLEL );
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> renderTime = end - start;
        std::cout << "RayTracer works "<< renderTime.count() << " seconds" << std::endl;
        //10.1, 9.9, 8,
        rayTracer->getCanvas()->saveToPNG( "out.png" );
        if ( true ) {
            Denoiser::denoise( rayTracer->getCanvas()->getColorData(), rayTracer->getCanvas()->getNormalData(), rayTracer->getCanvas()->getAlbedoData(), rayTracer->getCanvas()->getW(), rayTracer->getCanvas()->getH() );
            rayTracer->getCanvas()->saveToPNG( "outDenoised.png" );
        }
    } Kokkos::finalize();
    lua_close(L);
    return 0;
}