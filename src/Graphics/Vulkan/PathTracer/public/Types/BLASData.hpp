//
// Created by igor on 6/12/26.
//

#ifndef COLLECTION_BLAS_HPP
#define COLLECTION_BLAS_HPP

#include <string>
#include <vector>
#include <glm/glm.hpp>

#include "Buffer.hpp"
#include "AccelerationStructure.hpp"
#include "SharedTypes.h"

namespace crv::graphics::vulkan {
    struct BLASData {
        using GPU = MeshGPU;
        [[nodiscard]] GPU gpu(VkDevice device) const;
        [[nodiscard]] static std::vector<GPU> gpu(VkDevice device, const std::vector<BLASData>& data);

        Buffer                vertexBuffer = CRV_NULL_HANDLE;
        Buffer                indexBuffer  = CRV_NULL_HANDLE;
        Buffer                aliasBuffer  = CRV_NULL_HANDLE; // built only for emissive meshes
        AccelerationStructure blas         = CRV_NULL_HANDLE;
        uint32_t              indexCount   = 0;
        float                 area         = 0;
        uint32_t              modelIndex   = 0;
        std::string           meshName{};
        std::vector<float>    triAreas{};
        glm::vec3             aabbMin      = glm::vec3(0.0f);
        glm::vec3             aabbMax      = glm::vec3(0.0f);
    };

    inline BLASData::GPU BLASData::gpu(VkDevice device) const {
        return {
            .vertices = vertexBuffer.deviceAddress(device),
            .indices = indexBuffer.deviceAddress(device),
            .indexCount = indexCount,
            .area = area,
            .aliasAddress = aliasBuffer.get() != VK_NULL_HANDLE ? aliasBuffer.deviceAddress(device) : 0
        };
    }

    inline std::vector<BLASData::GPU> BLASData::gpu(VkDevice device, const std::vector<BLASData>& data) {
        std::vector<GPU> res{};
        res.reserve(data.size());
        for (const auto& blas: data) res.push_back(blas.gpu(device));
        return res;
    }
}

#endif //COLLECTION_BLAS_HPP