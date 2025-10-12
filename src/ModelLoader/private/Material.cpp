//
// Created by auser on 6/12/25.
//

#include "Material.hpp"

namespace crv::model {
    aiTextureType Texture::toAssimpType(Type type) {
        return toAssimpTypeMap[type];
    }
}