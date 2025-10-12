#ifndef COLLECTION_OBJLOADER_H
#define COLLECTION_OBJLOADER_H
#include "HittableList.h"
#include "Material.h"

namespace OBJLoader {
    bool load( const std::string& path, HittableList* world, Material* material = nullptr );
}


#endif //COLLECTION_OBJLOADER_H
