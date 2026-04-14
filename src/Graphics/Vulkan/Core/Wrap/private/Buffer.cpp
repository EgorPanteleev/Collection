//
// Created by igor on 4/14/26.
//

#include "Buffer.hpp"
#include "CommandPool.hpp"
#include "CommandBuffers.hpp"
#include "Message.hpp"
#include "Utils.hpp"

#include <stdexcept>
#include <cstring>

namespace crv::graphics::vulkan {
    Buffer::Buffer(const BufferCreateInfo &info): mAllocator(info.allocator), mSize(info.size) {
        std::tie(mHandle, mAllocation) = createBuffer(info);
    }

    void Buffer::destroy() {
        if (mHandle == VK_NULL_HANDLE or mAllocation == VK_NULL_HANDLE) return;
        vmaDestroyBuffer(mAllocator, mHandle, mAllocation);
        mAllocation = VK_NULL_HANDLE;
        mAllocator = VK_NULL_HANDLE;
    }

    std::tuple<VkBuffer, VmaAllocation> Buffer::createBuffer(const BufferCreateInfo& info) {
        VkBuffer buffer;
        VmaAllocation allocation;
        const VkBufferCreateInfo bufferInfo {
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = info.size,
            .usage = info.bufferUsage,
            .sharingMode = info.sharingMode
        };
        const VmaAllocationCreateInfo allocInfo {
            .usage = info.memoryUsage,
            .preferredFlags = 0
        };

        if (vmaCreateBuffer(info.allocator, &bufferInfo, &allocInfo, &buffer, &allocation, nullptr) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create buffer with VMA");
        }
        return {buffer, allocation};
    }

    void Buffer::copy(const CopyDataToCPUBufferInfo& info) {
        void* stagingData;
        vmaMapMemory(info.allocator, info.allocation, &stagingData);
        memcpy(stagingData, info.data, info.size);
        vmaUnmapMemory(info.allocator, info.allocation);
    }

    void Buffer::copy(const CopyCPUBufferToDataInfo& info) {
        void* mapped;
        vmaMapMemory(info.allocator, info.allocation, &mapped);
        memcpy(info.data, mapped, info.size);
        vmaUnmapMemory(info.allocator, info.allocation);
    }

    void Buffer::copy(const CopyBufferToBufferInfo& info) {
        auto [commandPool, commandBuffers] = beginCommandBuffer(info.device, info.queueFamilyIndex);
        VkCommandBuffer commandBuffer = (*commandBuffers)[0];
        const VkBufferCopy copyRegion{
            .srcOffset = 0,
            .dstOffset = 0,
            .size = info.size,
        };
        vkCmdCopyBuffer(commandBuffer, info.srcBuffer, info.dstBuffer, 1, &copyRegion);
        endCommandBuffer(commandPool, commandBuffers, info.queue);
    }

    void Buffer::copy(const CopyDataToGPUBufferInfo& info) {
        const BufferCreateInfo createInfo {
            .allocator = info.allocator,
            .size = info.size,
            .bufferUsage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .memoryUsage = VMA_MEMORY_USAGE_CPU_ONLY
        };
        auto [stagingBuffer, stagingAllocation] = createBuffer(createInfo);
        const CopyDataToCPUBufferInfo copyDataToCPUBufferInfo {
            .data = info.data,
            .size = info.size,
            .allocator = info.allocator,
            .allocation = stagingAllocation
        };
        copy(copyDataToCPUBufferInfo);
        const CopyBufferToBufferInfo copyBufferToBufferInfo {
            .srcBuffer = stagingBuffer,
            .dstBuffer = info.buffer,
            .size = info.size,
            .device = info.device,
            .queueFamilyIndex = info.queueFamilyIndex,
            .queue = info.queue
        };
        copy(copyBufferToBufferInfo);
        vmaDestroyBuffer(info.allocator, stagingBuffer, stagingAllocation);
    }

    void Buffer::copy(const CopyGPUBufferToDataInfo& info) {
        const BufferCreateInfo createInfo {
            .allocator = info.allocator,
            .size = info.size,
            .bufferUsage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .memoryUsage = VMA_MEMORY_USAGE_GPU_TO_CPU
        };
        auto [stagingBuffer, stagingAllocation] = createBuffer(createInfo);
        const CopyBufferToBufferInfo copyBufferToBufferInfo {
            .srcBuffer = info.buffer,
            .dstBuffer = stagingBuffer,
            .size = info.size,
            .device = info.device,
            .queueFamilyIndex = info.queueFamilyIndex,
            .queue = info.queue
        };
        copy(copyBufferToBufferInfo);

        const CopyCPUBufferToDataInfo copyCPUBufferToDataInfo {
            .data = info.data,
            .size = info.size,
            .allocator = info.allocator,
            .allocation = stagingAllocation
        };
        copy(copyCPUBufferToDataInfo);
        vmaDestroyBuffer(info.allocator, stagingBuffer, stagingAllocation);
    }
}
