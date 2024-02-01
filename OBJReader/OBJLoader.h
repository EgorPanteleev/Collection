#pragma once
#include "OBJShape.h"

class OBJLoader {
public:
    OBJLoader( const std::string& path, OBJShape* target );
    static bool load( const std::string& path, OBJShape* target );
};

