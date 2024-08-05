#pragma once
#include "Mesh.h"
#include <cstring>
class OBJLoader {
public:
    OBJLoader( const std::string& path, Mesh* target );
    static bool load( const std::string& path, Mesh* target );
};

