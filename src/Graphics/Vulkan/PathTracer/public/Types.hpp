//
// Created by igor on 6/8/26.
//

#ifndef COLLECTION_TYPES_HPP
#define COLLECTION_TYPES_HPP

#include "AccelerationStructure.hpp"
#include "Context.hpp"

#include <glm/gtx/quaternion.hpp>

namespace crv::graphics::vulkan {
    using SSBOInfo = std::tuple<void*, uint32_t, Buffer&>;
    using UBOInfo  = std::tuple<uint32_t, Buffer&>;

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

    struct BLASEntry {
        Buffer                vertexBuffer = CRV_NULL_HANDLE;
        Buffer                indexBuffer  = CRV_NULL_HANDLE;
        AccelerationStructure blas         = CRV_NULL_HANDLE;
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
}

#endif //COLLECTION_TYPES_HPP