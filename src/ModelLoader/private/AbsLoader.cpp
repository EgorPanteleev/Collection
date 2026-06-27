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
            case gli::FORMAT_RGBA_BP_UNORM_BLOCK16:
                return Texture::BC7_UNORM;
            case gli::FORMAT_RGBA_BP_SRGB_BLOCK16:
                return Texture::BC7_SRGB;
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
            int layer = 0; int face = 0; int level = 0;
            const size_t size = tex.size(level);
            void* data = std::malloc(size);
            memcpy(data, tex.data(layer, face, level), size);
            texture.mDataByLevel.push_back({data, (uint32_t) tex.extent(level).x,
                                         (uint32_t) tex.extent(level).y});
            texture.mFormat = toTextureFormat( tex.format() );
        }
        return texture;
    }

    static uint16_t floatToHalf(float value) {
        uint32_t bits;
        std::memcpy(&bits, &value, sizeof(bits));
        const uint32_t sign = (bits >> 16) & 0x8000u;
        int32_t exponent = static_cast<int32_t>((bits >> 23) & 0xffu) - 127 + 15;
        const uint32_t mantissa = bits & 0x7fffffu;
        if (exponent <= 0)  return static_cast<uint16_t>(sign);
        if (exponent >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
        return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10) | (mantissa >> 13));
    }

    Texture AbsLoader::loadSkybox(const std::string& path) {
        Texture texture;
        int width, height, channels;
        float* pixels = stbi_loadf(path.c_str(), &width, &height, &channels, STBI_rgb_alpha);
        if (!pixels) throw std::runtime_error("Failed to load skybox: " + path);
        const size_t count = static_cast<size_t>(width) * height * 4;
        auto* halfData = static_cast<uint16_t*>(std::malloc(count * sizeof(uint16_t)));
        for (size_t i = 0; i < count; ++i) halfData[i] = floatToHalf(pixels[i]);
        stbi_image_free(pixels);
        texture.mFormat = Texture::R16G16B16A16_SFLOAT;
        texture.mDataByLevel.emplace_back(static_cast<void*>(halfData),
                                          static_cast<uint32_t>(width), static_cast<uint32_t>(height));
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
