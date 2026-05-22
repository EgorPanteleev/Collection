//
// Created by igor on 5/10/26.
//

#ifndef COLLECTION_TYPES_HPP
#define COLLECTION_TYPES_HPP

#include <vector>
#include "CoreUtils.hpp"
#include "GPUTypes.hpp"
#include "Context.hpp"
#include "Buffer.hpp"

namespace crv::graphics::vulkan {
    using SSBOInfo = std::tuple<void*, uint32_t, Buffer&>;
    using UBOInfo = std::tuple<uint32_t, Buffer&>;
    using UIVec2 = glm::vec<2, uint32_t>;

    struct TexturesByType {
        Texture& operator[](const int type) { return mTexturesByType[type]; }
        std::array<Texture, cm::Texture::UNKNOWN> mTexturesByType{};
    };

    struct GBuffer {
        Image     colorImage{};
        ImageView colorView{};
        Image     depthImage{};
        ImageView depthView{};
        Image     normalImage{};
        ImageView normalView{};
        Image     selectedInstanceImage{};
        ImageView selectedInstanceView{};
        Sampler   sampler{};
        Sampler   intSampler{};
    };

    struct MeshData {
        uint32_t baseVertex    = 0;
        uint32_t baseIndex     = 0;
        uint32_t indexCount    = 0;
        uint32_t baseInstance = 0;
        uint32_t instanceCount = 0;
    };

    struct SSBOData: std::vector<SSBOInfo> {
        SSBOData() = default;
        template <typename Type>
        void add(const std::vector<Type>& dataBuffer, Buffer& buffer) {
            emplace_back(const_cast<Type*>(dataBuffer.data()), dataBuffer.size() * sizeof(Type), buffer);
        }

        void createAll(Context* context, QueueFamilyType familyType) const {
            for (const auto& ssboInfo: *this) {
                createSSBO(context->allocator(), std::get<1>(ssboInfo), std::get<2>(ssboInfo));
                copyDataToBuffer(context, familyType, std::get<0>(ssboInfo), std::get<1>(ssboInfo), std::get<2>(ssboInfo));
            }
        }
    };

    struct UBOData: std::vector<UBOInfo> {
        UBOData() = default;
        template <typename Type>
        void add(Buffer& buffer) {
            emplace_back(sizeof(Type), buffer);
        }

        void createAll(Context* context, QueueFamilyType familyType) const {
            for (const auto& uboInfo: *this) {
                createUBO(context->allocator(), std::get<0>(uboInfo), std::get<1>(uboInfo));
            }
        }
    };
}

#endif //COLLECTION_TYPES_HPP