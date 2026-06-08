//
// Created by igor on 4/14/26.
//

#include "Buffer.hpp"
#include "CommandPool.hpp"
#include "CommandBuffers.hpp"
#include "Message.hpp"
#include "CoreUtils.hpp"

#include <stdexcept>
#include <cstring>

namespace crv::graphics::vulkan {
    Buffer::Buffer(const BufferCreateInfo &info): mAllocator(info.allocator), mSize(info.size) {
        VmaAllocation allocation;
        std::tie(mHandle, allocation) = createBuffer(info);
        mAllocation = Allocation(allocation);
    }

    Buffer::Buffer(Buffer&& other) noexcept {
        mAllocator = other.mAllocator;
        mAllocation = std::move(other.mAllocation);
        mSize = other.mSize;
        mHandle = other.mHandle;
        other.mHandle = VK_NULL_HANDLE;
        other.mAllocator = VK_NULL_HANDLE;
        other.mAllocation = {};
        other.mSize = 0;
    }

    void Buffer::destroy() {
        if (mHandle == VK_NULL_HANDLE) return;
        vmaDestroyBuffer(mAllocator, mHandle, mAllocation.get());
        mHandle = VK_NULL_HANDLE;
        mAllocator = VK_NULL_HANDLE;
        mAllocation.destroy();
    }

    VkDeviceAddress Buffer::deviceAddress(VkDevice device) const {
        const VkBufferDeviceAddressInfo addressInfo{
            .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
            .buffer = mHandle
        };
        return vkGetBufferDeviceAddress(device, &addressInfo);
    }

    void* Buffer::map() const {
        void* ptr;
        vmaMapMemory(mAllocator, mAllocation.get(), &ptr);
        return ptr;
    }

    void Buffer::unmap() const {
        vmaUnmapMemory(mAllocator, mAllocation.get());
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
            .flags = info.allocFlags,
            .usage = info.memoryUsage,
            .preferredFlags = 0
        };

        VkResult result;
        if (info.minAlignment > 1) {
            result = vmaCreateBufferWithAlignment(info.allocator, &bufferInfo, &allocInfo,
                info.minAlignment, &buffer, &allocation, nullptr);
        } else {
            result = vmaCreateBuffer(info.allocator, &bufferInfo, &allocInfo, &buffer, &allocation, nullptr);
        }

        if (result != VK_SUCCESS) {
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
        memcpy(const_cast<void*>(info.data), mapped, info.size);
        vmaUnmapMemory(info.allocator, info.allocation);
    }

    void Buffer::copy(const CopyBufferToBufferInfo& info) {
        auto [commandBuffer, cmdData] = beginCommandBuffer(info.device, info.queueFamilyIndex);
        const VkBufferCopy copyRegion{
            .srcOffset = 0,
            .dstOffset = 0,
            .size = info.size,
        };
        vkCmdCopyBuffer(commandBuffer, info.srcBuffer, info.dstBuffer, 1, &copyRegion);
        endCommandBuffer(cmdData, info.queue);
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
