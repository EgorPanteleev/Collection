//
// Created by igor on 6/10/26.
//

#ifndef COLLECTION_INSTANCEDATA_HPP
#define COLLECTION_INSTANCEDATA_HPP

#include "CoreUtils.hpp"

#include <string>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/matrix_decompose.hpp>
#include "SharedTypes.h"

namespace crv::graphics::vulkan {
    struct Transform {
        [[nodiscard]] glm::mat4 matrix() const;

        glm::vec3 position {0.0f};
        glm::quat rotation {1.0f, 0.0f, 0.0f, 0.0f};
        glm::vec3 scale    {1.0f};
    };

    struct InstanceData {
        static constexpr uint32_t NO_MESH = UINT32_MAX;
        using GPU = InstanceGPU;
        using AS = VkAccelerationStructureInstanceKHR;
        [[nodiscard]] bool isGroup() const { return meshIndex == NO_MESH; }
        [[nodiscard]] GPU gpu() const;
        [[nodiscard]] AS vkAS(uint32_t customIndex, VkDeviceAddress blasAddress) const;
        [[nodiscard]] static std::vector<GPU> gpu(const std::vector<InstanceData>& instances);

        std::string name{};
        std::string meshName{};
        Transform   transform{};
        glm::mat4   world          = glm::mat4(1.0f);
        int32_t     parentIndex    = -1;
        uint32_t    meshIndex      = 0;
        uint32_t    materialIndex  = 0;
        uint32_t    indexCount     = 0;
    };

    inline Transform decomposeTransform(const glm::mat4& matrix) {
        glm::vec3 scale(1.0f), translation(0.0f), skew(0.0f);
        glm::vec4 perspective(0.0f, 0.0f, 0.0f, 1.0f);
        glm::quat rotation(1.0f, 0.0f, 0.0f, 0.0f);
        if (glm::decompose(matrix, scale, rotation, translation, skew, perspective))
            return { translation, rotation, scale };

        translation = glm::vec3(matrix[3]);
        glm::vec3 axes[3] = { glm::vec3(matrix[0]), glm::vec3(matrix[1]), glm::vec3(matrix[2]) };
        scale = glm::vec3(glm::length(axes[0]), glm::length(axes[1]), glm::length(axes[2]));
        const glm::vec3 fallbackAxis[3] = { {1,0,0}, {0,1,0}, {0,0,1} };
        for (int i = 0; i < 3; ++i)
            axes[i] = scale[i] > 1e-8f ? axes[i] / scale[i] : fallbackAxis[i];
        glm::mat3 basis(axes[0], axes[1], axes[2]);
        if (glm::determinant(basis) < 0.0f) { basis[0] = -basis[0]; scale.x = -scale.x; }
        rotation = glm::normalize(glm::quat_cast(basis));
        return { translation, rotation, scale };
    }

    inline glm::mat4 Transform::matrix() const {
        const glm::mat4 T = glm::translate(glm::mat4(1.0f), position);
        const glm::mat4 R = glm::toMat4(rotation);
        const glm::mat4 S = glm::scale(glm::mat4(1.0f), scale);
        return T * R * S;
    }

    inline InstanceData::GPU InstanceData::gpu() const {
        return {
            .model = world,
            .meshIndex = meshIndex,
            .materialIndex = materialIndex,
            .lightArea = 0.0f,
        };
    }

    inline InstanceData::AS InstanceData::vkAS(const uint32_t customIndex, const VkDeviceAddress blasAddress) const {
        return {
            .transform = toVkTransform(world),
            .instanceCustomIndex = customIndex,
            .mask = 0xFF,
            .instanceShaderBindingTableRecordOffset = 0,
            .flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR,
            .accelerationStructureReference = blasAddress
        };
    }

    inline std::vector<InstanceData::GPU> InstanceData::gpu(const std::vector<InstanceData>& instances) {
        std::vector<GPU> res{};
        res.reserve(instances.size());
        for (const auto& instance: instances) res.push_back(instance.gpu());
        return res;
    }
}

#endif //COLLECTION_INSTANCEDATA_HPP