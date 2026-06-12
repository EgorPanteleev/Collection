//
// Created by igor on 6/10/26.
//

#ifndef COLLECTION_INSTANCEDATA_HPP
#define COLLECTION_INSTANCEDATA_HPP

#include <string>
#include <glm/gtx/quaternion.hpp>

namespace crv::graphics::vulkan {
    struct Transform {
        [[nodiscard]] glm::mat4 matrix() const;

        glm::vec3 position {0.0f};
        glm::quat rotation {1.0f, 0.0f, 0.0f, 0.0f};
        glm::vec3 scale    {1.0f};
    };

    struct InstanceGPU {
        uint32_t meshIndex     = 0;
        uint32_t materialIndex = 0;
    };

    struct InstanceData {
        using GPU = InstanceGPU;
        using AS = VkAccelerationStructureInstanceKHR;
        [[nodiscard]] GPU gpu() const;
        [[nodiscard]] AS vkAS(uint32_t customIndex, VkDeviceAddress blasAddress) const;
        [[nodiscard]] static std::vector<GPU> gpu(const std::vector<InstanceData>& instances);

        std::string name{};
        std::string meshName{};
        Transform   transform{};
        uint32_t    meshIndex      = 0;
        uint32_t    materialIndex  = 0;
        uint32_t    indexCount     = 0;
    };

    inline glm::mat4 Transform::matrix() const {
        const glm::mat4 T = glm::translate(glm::mat4(1.0f), position);
        const glm::mat4 R = glm::toMat4(rotation);
        const glm::mat4 S = glm::scale(glm::mat4(1.0f), scale);
        return T * R * S;
    }

    inline InstanceData::GPU InstanceData::gpu() const {
        return {
            .meshIndex = meshIndex,
            .materialIndex = materialIndex,
        };
    }

    inline InstanceData::AS InstanceData::vkAS(const uint32_t customIndex, const VkDeviceAddress blasAddress) const {
        return {
            .transform = toVkTransform(transform.matrix()),
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