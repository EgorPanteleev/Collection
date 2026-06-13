//
// Created by igor on 6/12/26.
//

#ifndef COLLECTION_BLAS_HPP
#define COLLECTION_BLAS_HPP

#include "Buffer.hpp"
#include "AccelerationStructure.hpp"

namespace crv::graphics::vulkan {
    struct BLASDataGPU {
        VkDeviceAddress vertexAddress = 0;
        VkDeviceAddress indexAddress  = 0;
        uint32_t        indexCount    = 0;
        float           area          = 0;
    };

    struct BLASData {
        using GPU = BLASDataGPU;
        [[nodiscard]] GPU gpu(VkDevice device) const;
        [[nodiscard]] static std::vector<GPU> gpu(VkDevice device, const std::vector<BLASData>& data);

        Buffer                vertexBuffer = CRV_NULL_HANDLE;
        Buffer                indexBuffer  = CRV_NULL_HANDLE;
        AccelerationStructure blas         = CRV_NULL_HANDLE;
        uint32_t              indexCount   = 0;
        float                 area         = 0;
    };

    inline BLASData::GPU BLASData::gpu(VkDevice device) const {
        return {
            .vertexAddress = vertexBuffer.deviceAddress(device),
            .indexAddress = indexBuffer.deviceAddress(device),
            .indexCount = indexCount,
            .area = area
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