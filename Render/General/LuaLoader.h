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

float loadNumber( lua_State* L, const std::string& fieldName );

float loadNumber( lua_State* L );

Vector<float> loadTable( lua_State* L, const std::string& fieldName );

Vector<float> loadTable( lua_State* L );

Vector3f loadVector3f( lua_State* L, const std::string& fieldName );

Vector3f loadVector3f( lua_State* L );

Canvas* loadCanvas( lua_State* L );

Material loadMaterial( lua_State* L );

bool loadScene( lua_State* L, Scene* scene );

bool loadMeshes( lua_State* L, Scene* scene );

bool loadSpheres( lua_State* L, Scene* scene );

Camera* loadCamera( lua_State* L, int w, int h  );

Vector<float> loadSettings( lua_State* L );

#endif //COLLECTION_LUALOADER_H
