#pragma once
#include "Mesh.h"
#include "GroupOfMeshes.h"
#include <cstring>
class OBJLoader {
public:
    OBJLoader( const std::string& path, Mesh* target );
    OBJLoader( const std::string& path, GroupOfMeshes* target );
    static bool load( const std::string& path, Mesh* target );
    static bool load( const std::string& path, GroupOfMeshes* target );
};

