//
// Created by igor on 6/8/26.
//

#ifndef COLLECTION_ACCELERATIONSTRUCTURE_HPP
#define COLLECTION_ACCELERATIONSTRUCTURE_HPP

#include "Buffer.hpp"
#include "Instance.hpp"

namespace crv::graphics::vulkan {
    struct BLASCreateInfo {
        VkCommandBuffer  commandBuffer = VK_NULL_HANDLE;
        VkDevice         device        = VK_NULL_HANDLE;
        VkPhysicalDevice physicalDevice  = VK_NULL_HANDLE;
        VmaAllocator     allocator     = VK_NULL_HANDLE;
        VkDeviceAddress  vertexAddress = 0;
        VkDeviceSize     vertexStride  = 0;
        uint32_t         vertexCount   = 0;
        VkDeviceAddress  indexAddress  = 0;
        uint32_t         indexCount    = 0;
    };

    struct TLASCreateInfo {
        VkCommandBuffer  commandBuffer   = VK_NULL_HANDLE;
        VkDevice         device          = VK_NULL_HANDLE;
        VkPhysicalDevice physicalDevice  = VK_NULL_HANDLE;
        VmaAllocator     allocator       = VK_NULL_HANDLE;
        VkDeviceAddress  instanceAddress = 0;
        uint32_t         instanceCount   = 0;
    };

    struct ASCreateInfo {
        VkAccelerationStructureTypeKHR       type           = VK_ACCELERATION_STRUCTURE_TYPE_MAX_ENUM_KHR;
        VkBuildAccelerationStructureFlagsKHR flags          = 0;
        VkCommandBuffer                      commandBuffer  = VK_NULL_HANDLE;
        VkPhysicalDevice                     physicalDevice = VK_NULL_HANDLE;
        VmaAllocator                         allocator      = VK_NULL_HANDLE;
        uint32_t                             primitiveCount = 0;
    };

    struct TLASUpdateInfo {
        VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
        uint32_t        instanceCount = 0;
    };

    class AccelerationStructure: public DefaultWrapper<VkAccelerationStructureKHR> {
    public:
        using DefaultWrapper::DefaultWrapper;
        explicit AccelerationStructure(const BLASCreateInfo& info);
        explicit AccelerationStructure(const TLASCreateInfo& info);
        AccelerationStructure(AccelerationStructure&&) = default;
        AccelerationStructure& operator=(AccelerationStructure&&) = default;
        ~AccelerationStructure() override { AccelerationStructure::destroy(); }
        VkDeviceAddress deviceAddress() const;
        void update(const TLASUpdateInfo& info);
        void destroy() override;
    protected:
        void create(const ASCreateInfo& info);
        void create(const BLASCreateInfo& info);
        void create(const TLASCreateInfo& info);

        VkDevice                           mDevice        = VK_NULL_HANDLE;
        Buffer                             mBuffer        = CRV_NULL_HANDLE;
        Buffer                             mScratchBuffer = CRV_NULL_HANDLE;
        VkAccelerationStructureGeometryKHR mGeometry{};
    };
}

#endif //COLLECTION_ACCELERATIONSTRUCTURE_HPP