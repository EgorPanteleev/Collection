//
// Created by igor on 5/10/26.
//

#ifndef COLLECTION_TYPES_HPP
#define COLLECTION_TYPES_HPP

#include <vector>
#include <glm/gtx/quaternion.hpp>

#include "CoreUtils.hpp"
#include "Context.hpp"
#include "Buffer.hpp"
#include "Texture.hpp"

namespace crv::graphics::vulkan {
    using SSBOInfo = std::tuple<void*, uint32_t, Buffer&>;
    using UBOInfo  = std::tuple<uint32_t, Buffer&>;
    using UIVec2   = glm::vec<2, uint32_t>;

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

    struct Transform {
        glm::vec3 position {0.0f};
        glm::quat rotation {1.0f, 0.0f, 0.0f, 0.0f};
        glm::vec3 scale    {1.0f};

        [[nodiscard]] glm::mat4 matrix() const {
            glm::mat4 T = glm::translate(glm::mat4(1.0f), position);
            glm::mat4 R = glm::toMat4(rotation);
            glm::mat4 S = glm::scale(glm::mat4(1.0f), scale);
            return T * R * S;
        }
    };

    struct MeshInstance {
        std::string name{};
        std::string meshName{};
        Transform transform{};
        uint32_t baseNode = 0;
        uint32_t baseTri = 0; //can be removed
        uint32_t texIndex = 0;
    };

    struct SSBOData: std::vector<SSBOInfo> {
        SSBOData() = default;
        template <typename Type>
        void add(const std::vector<Type>& dataBuffer, Buffer& buffer) { emplace_back(const_cast<Type*>(dataBuffer.data()), dataBuffer.size() * sizeof(Type), buffer); }
        void createAll(Context* context, QueueFamilyType familyType) const;
    };

    struct UBOData: std::vector<UBOInfo> {
        UBOData() = default;
        template <typename Type>
        void add(Buffer& buffer) { emplace_back(sizeof(Type), buffer); }
        void createAll(Context* context) const;
    };
}

#endif //COLLECTION_TYPES_HPP