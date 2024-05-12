#pragma once
#include "TriangularMesh.h"
#include <cstring>
class OBJLoader {
public:
    OBJLoader( const std::string& path, TriangularMesh* target );
    static bool load( const std::string& path, TriangularMesh* target );
};

