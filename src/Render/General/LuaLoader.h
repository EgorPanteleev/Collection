//
// Created by auser on 8/5/24.
//

#ifndef COLLECTION_LUALOADER_H
#define COLLECTION_LUALOADER_H
#include <iostream>
#include "Camera.h"
#include "Canvas.h"
#include "Scene.h"
#include "CubeMesh.h"
#include "PointLight.h"
#include "SpotLight.h"
#include "cstdlib"

extern "C" {
#include "lua.h"
#include "lualib.h"
#include "lauxlib.h"
}

double loadNumber( lua_State* L, const std::string& fieldName );

double loadNumber( lua_State* L );

Vector<double> loadTable( lua_State* L, const std::string& fieldName );

Vector<double> loadTable( lua_State* L );

Vec3d loadVec3d( lua_State* L, const std::string& fieldName );

Vec3d loadVec3d( lua_State* L );

Canvas* loadCanvas( lua_State* L );

Material loadMaterial( lua_State* L );

bool loadScene( lua_State* L, Scene* scene );

bool loadMeshes( lua_State* L, Scene* scene );

bool loadSpheres( lua_State* L, Scene* scene );

Camera* loadCamera( lua_State* L, int w, int h  );

Vector<double> loadSettings( lua_State* L );

#endif //COLLECTION_LUALOADER_H
