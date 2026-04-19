//
// Created by igor on 4/14/26.
//

#ifndef COLLECTION_BUFFER_HPP
#define COLLECTION_BUFFER_HPP

#include "DefaultWrapper.hpp"
#include "Allocation.hpp"

#include <vk_mem_alloc.h>

namespace crv::graphics::vulkan {
    struct BufferCreateInfo {
        VmaAllocator allocator = VK_NULL_HANDLE;
        VkDeviceSize size = 0;
        VkBufferUsageFlags bufferUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        VkSharingMode sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VmaMemoryUsage memoryUsage = VMA_MEMORY_USAGE_UNKNOWN;
    };

    struct CopyDataToCPUBufferInfo {
        void* data = nullptr;
        VkDeviceSize size = 0;
        VmaAllocator allocator = VK_NULL_HANDLE;
        VmaAllocation allocation = VK_NULL_HANDLE;
    };

    struct CopyCPUBufferToDataInfo {
        void* data = nullptr;
        VkDeviceSize size = 0;
        VmaAllocator allocator = VK_NULL_HANDLE;
        VmaAllocation allocation = VK_NULL_HANDLE;
    };

    struct CopyBufferToBufferInfo {
        VkBuffer srcBuffer = VK_NULL_HANDLE;
        VkBuffer dstBuffer = VK_NULL_HANDLE;
        VkDeviceSize size = 0;
        VkDevice device = VK_NULL_HANDLE;
        uint32_t queueFamilyIndex = 0;
        VkQueue queue = VK_NULL_HANDLE;
    };

    struct CopyDataToGPUBufferInfo {
        void* data = nullptr;
        VkDeviceSize size = 0;
        VmaAllocator allocator = VK_NULL_HANDLE;
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDevice device = VK_NULL_HANDLE;
        uint32_t queueFamilyIndex = 0;
        VkQueue queue = VK_NULL_HANDLE;
    };

    struct CopyGPUBufferToDataInfo {
        void* data = nullptr;
        VkDeviceSize size = 0;
        VmaAllocator allocator = VK_NULL_HANDLE;
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDevice device = VK_NULL_HANDLE;
        uint32_t queueFamilyIndex = 0;
        VkQueue queue = VK_NULL_HANDLE;
    };

    class Buffer: public DefaultWrapper<VkBuffer> {
    public:
        using DefaultWrapper::DefaultWrapper;
        explicit Buffer(const BufferCreateInfo& info);
        explicit Buffer(Buffer&&);
        Buffer& operator=(Buffer&&) = default;
        ~Buffer() override { Buffer::destroy(); }
        void destroy() override;
        [[nodiscard]] VmaAllocation allocation() const { return mAllocation.get(); }
        [[nodiscard]] VkDeviceSize size() const { return mSize; }
        static std::tuple<VkBuffer, VmaAllocation> createBuffer(const BufferCreateInfo& info);
        static void copy(const CopyDataToCPUBufferInfo& info);
        static void copy(const CopyCPUBufferToDataInfo& info);
        static void copy(const CopyBufferToBufferInfo& info);
        static void copy(const CopyDataToGPUBufferInfo& info);
        static void copy(const CopyGPUBufferToDataInfo& info);
    protected:
        VmaAllocator mAllocator = VK_NULL_HANDLE;
        Allocation mAllocation{};
        VkDeviceSize mSize = 0;
    };
}

#endif //COLLECTION_BUFFER_HPP