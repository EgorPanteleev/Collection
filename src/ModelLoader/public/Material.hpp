//
// Created by auser on 6/12/25.
//

#ifndef VULKAN_MODELMATERIAL_H
#define VULKAN_MODELMATERIAL_H

#include <string>
#include <array>
#include <vector>
#include <map>
#include <assimp/material.h>
#include <glm/glm.hpp>

namespace crv::model {
    struct Texture {
        enum Type {
            DIFFUSE = 0,
            SPECULAR = 1,
            SHININESS = 2,
            AMBIENT = 3,
            NORMAL = 4,
            UNKNOWN = 5
        };

        enum Format {
            R8G8B8A8_SRGB = 0,
            R8G8B8A8_UNORM = 1,
            BC1_UNORM = 2,
            BC3_UNORM = 3,
            BC5_UNORM = 4,
            UNDEFINED = 5,
        };

        Texture() : mDataByLevel() {}

        bool empty() const { return mDataByLevel.empty(); }

        static aiTextureType toAssimpType(Type type);

        struct LevelData {
            void* data;
            uint32_t width;
            uint32_t height;
        };
        std::vector<LevelData> mDataByLevel;
        Format mFormat;
    };

    static std::map<Texture::Type, aiTextureType> toAssimpTypeMap{
            {Texture::DIFFUSE,   aiTextureType_DIFFUSE},
            {Texture::SPECULAR,  aiTextureType_SPECULAR},
            {Texture::SHININESS, aiTextureType_SHININESS},
            {Texture::AMBIENT,   aiTextureType_AMBIENT},
            {Texture::NORMAL,    aiTextureType_NORMALS},
            {Texture::UNKNOWN,   aiTextureType_UNKNOWN}
    };

    struct Material {
        Material() : mName(), ambientColor(0), diffuseColor(0), specularColor(0),
                     mTransparencyFactor(1), mAlphaTest(0) {}

        std::string mName;

        glm::vec4 ambientColor;
        glm::vec4 diffuseColor;
        glm::vec4 specularColor;

        float mTransparencyFactor;
        float mAlphaTest;

        std::array<Texture, Texture::UNKNOWN> mTextures;
    };
}
#endif //VULKAN_MODELMATERIAL_H
