//
// Created by auser on 5/5/25.
//

#include "AbsLoader.hpp"
#include "Message.hpp"

#include <unordered_map>
#include <filesystem>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#include <gli/gli.hpp>

namespace fs = std::filesystem;

namespace crv::model {
    AbsLoader::AbsLoader(std::string modelPath) : mModelPath(std::move(modelPath)) {}

    static glm::vec3 min(const glm::vec3 &v1, const glm::vec3 &v2) {
        return {
                std::min(v1.x, v2.x),
                std::min(v1.y, v2.y),
                std::min(v1.z, v2.z)
        };
    }

    static glm::vec3 max(const glm::vec3 &v1, const glm::vec3 &v2) {
        return {
                std::max(v1.x, v2.x),
                std::max(v1.y, v2.y),
                std::max(v1.z, v2.z)
        };
    }

    void AbsLoader::computeBBox() {
        for (const auto &vert: vertices()) {
            mBBox.min = min(mBBox.min, vert.pos);
            mBBox.max = max(mBBox.max, vert.pos);
        }
    }

    void AbsLoader::clear() {
        mMeshes.clear();
        mMaterials.clear();
        mVertices.clear();
        mIndices.clear();
    }

    Texture::Format AbsLoader::toTextureFormat(Texture::Type modelTexType) {
        switch(modelTexType) {
            case Texture::Type::BASE_COLOR:
                return Texture::R8G8B8A8_SRGB;
            case Texture::Type::NORMAL:
            case Texture::Type::METAL_ROUGHNESS:
                return Texture::R8G8B8A8_UNORM;
            default:
                INFO << "ID: " << modelTexType;
                throw std::runtime_error("Unsupported model texture format!");
        }
    }

    Texture::Format AbsLoader::toTextureFormat(gli::texture::format_type gliFormat) {
        switch(gliFormat) {
            case gli::FORMAT_RGBA_DXT1_UNORM_BLOCK8:
                return Texture::BC1_UNORM;
            case gli::FORMAT_RGBA_DXT5_UNORM_BLOCK16:
                return Texture::BC3_UNORM;
            case gli::FORMAT_RG_ATI2N_UNORM_BLOCK16:
                return Texture::BC5_UNORM;
            default:
                INFO << "ID: " << gliFormat;
                throw std::runtime_error("Unsupported gli format!");
        }
    }

    Texture AbsLoader::loadTexture(const std::string& path, const Texture::Type type) {
        Texture texture;
        texture.mFormat = toTextureFormat(type);
        int width, height, channels;
        void* pixels = stbi_load(path.c_str(), &width, &height, &channels, STBI_rgb_alpha);
        if ( pixels ) {
            texture.mDataByLevel.emplace_back(pixels, width, height);
        } else {
            gli::texture tex = gli::load(path.c_str());
            if (tex.empty()) throw std::runtime_error("Failed to load texture!");
            texture.mDataByLevel.resize(tex.levels());
            int layer = 0; int face = 0;
            for (int level = 0; level < tex.levels(); ++level) {
                pixels = tex.data(layer, face, level);
                texture.mDataByLevel[level] = {pixels, (uint32_t) tex.extent(level).x,
                                             (uint32_t) tex.extent(level).y};
            }
            texture.mFormat = toTextureFormat( tex.format() );
        }
        return texture;
    }

    static glm::vec3 toEmptyColor(Texture::Type texType) {
        switch(texType) {
            case Texture::Type::BASE_COLOR:
                return {0.5, 0, 0};
            case Texture::Type::NORMAL:
                return {0.5, 0.5, 1};
            case Texture::Type::METAL_ROUGHNESS:
                return {0, 1, 0};
            default:
                INFO << "ID: " << texType;
                throw std::runtime_error("Unsupported model texture type!");
        }
    }

    Texture AbsLoader::colorTexture(const glm::vec3& color, Texture::Type texType) {
        Texture texture;
        texture.mFormat = toTextureFormat(texType);
        gli::texture2d tex(gli::FORMAT_RGBA8_UNORM_PACK8, {1,1}, 1);
        tex.clear(gli::packUnorm4x8(glm::vec4(color, 1)));
        const size_t size = tex.size();
        void* data = std::malloc(size); //leak here
        memcpy(data, tex.data(), size);
        texture.mDataByLevel.emplace_back(data, 1, 1);
        return texture;
    }

    Texture AbsLoader::emptyTexture(const Texture::Type texType) {
        return colorTexture(toEmptyColor(texType), texType);
    }
}
