//
// Created by auser on 12/31/24.
//

#ifndef COLLECTION_LUALOADER_H
#define COLLECTION_LUALOADER_H

#include "Vector.h"
#include "Vec3.h"
#include "Camera.h"

extern "C" {
#include "lua.h"
#include "lualib.h"
#include "lauxlib.h"
}


namespace Lua {
    lua_State* newState();

    bool open( lua_State* luaState, const std::string& filePath );

    void close( lua_State* luaState );

    void getGlobal( lua_State* luaState, const std::string& name );

    void pop( lua_State* luaState, int idx );

    bool isNumber( lua_State* luaState, int idx );

    bool isString( lua_State* luaState, int idx );

    bool isTable( lua_State* luaState, int idx );

    void getField( lua_State* luaState, int idx, const std::string& name );

    void pushInteger( lua_State* luaState, int idx );

    void pushNil( lua_State* luaState );

    double toNumber( lua_State* luaState, int idx );

    std::string toString( lua_State* luaState, int idx );

    double next( lua_State* luaState, int idx );

    size_t length( lua_State* luaState, int idx );

    void getElement( lua_State* luaState, int idxTable, int idxElement );

    bool loadString( lua_State* luaState, std::string& str );

    bool loadString( lua_State* luaState, const std::string& name, std::string& str );

    template <typename Type>
    bool loadNumber( lua_State* luaState, Type& number ) {
        if ( !isNumber( luaState, -1 ) ) {
            std::cout << "Failed to load number.\n";
            pop( luaState, 1);
            return false;
        }
        number = toNumber( luaState, -1 );
        pop( luaState, 1);
        return true;
    }

    template <typename Type>
    bool loadNumber( lua_State* luaState, const std::string& name, Type& number ) {
        getField( luaState, -1, name );
        if (!isNumber( luaState, -1 ) ) {
            std::cout << "Failed to load number.\n";
            pop( luaState, 1);
            return false;
        }
        number = toNumber( luaState, -1 );
        pop( luaState, 1);
        return true;
    }

    bool loadVec3d( lua_State* luaState, Vec3d& vec );

    bool loadVec3d( lua_State* luaState, const std::string& name, Vec3d& vec );

    bool loadRenderSettings( lua_State* luaState, Camera* cam );

    bool loadMaterial( lua_State* luaState, Material*& material );

    bool loadSpheres( lua_State* luaState, HittableList* world );

    bool loadWorld( lua_State* luaState, HittableList* world );
}

namespace LuaLoader {

    //---------

    double loadNumber( lua_State* L, const std::string& fieldName );

    double loadNumber( lua_State* L );

    Vector<double> loadTable( lua_State* L, const std::string& fieldName );

    Vector<double> loadTable( lua_State* L );

    Vec3d loadVec3d( lua_State* L, const std::string& fieldName );

    Vec3d loadVec3d( lua_State* L );
//
//    Canvas* loadCanvas( lua_State* L );
//
//    Material loadMaterial( lua_State* L );
//
//    bool loadScene( lua_State* L, Scene* scene );
//
//    bool loadMeshes( lua_State* L, Scene* scene );
//
//    bool loadSpheres( lua_State* L, Scene* scene );
//
//    Camera* loadCamera( lua_State* L, int w, int h  );
//
//    Vector<double> loadSettings( lua_State* L );
}

#endif //COLLECTION_LUALOADER_H
