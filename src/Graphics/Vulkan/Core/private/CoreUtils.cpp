//
// Created by igor on 4/14/26.
//

#include "CoreUtils.hpp"
#include <stdexcept>

namespace crv::graphics::vulkan {
    std::tuple<CommandPool*, CommandBuffers*> beginCommandBuffer(VkDevice device, const uint32_t queueFamilyIndex) {
        const CommandPoolCreateInfo poolCreateInfo {
            .device = device,
            .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
            .queueFamilyIndex = queueFamilyIndex
        };
        auto commandPool = new CommandPool(poolCreateInfo);

        const CommandBuffersCreateInfo createInfo {
            .device = device,
            .commandPool = commandPool->get(),
            .bufferCount = 1
        };
        auto commandBuffers = new CommandBuffers(createInfo);
        VkCommandBuffer commandBuffer = (*commandBuffers)[0];

        const VkCommandBufferBeginInfo beginInfo {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
        };
        vkBeginCommandBuffer(commandBuffer, &beginInfo);
        return {commandPool, commandBuffers};
    }

    void endCommandBuffer(CommandPool* commandPool, CommandBuffers* commandBuffers, VkQueue queue) {
        VkCommandBuffer commandBuffer = (*commandBuffers)[0];
        vkEndCommandBuffer(commandBuffer);
        const VkSubmitInfo submitInfo{
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .commandBufferCount = 1,
            .pCommandBuffers = &commandBuffer
        };
        vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
        vkQueueWaitIdle(queue);
        commandBuffers->destroy();
        commandPool->destroy();
    }

    void beginCommandBuffer(VkCommandBuffer commandBuffer) {
        constexpr VkCommandBufferBeginInfo beginInfo {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = 0,
            .pInheritanceInfo = nullptr
        };
        if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
            throw std::runtime_error("Failed to begin recording command buffer!");
        }
    }

    void endCommandBuffer(VkCommandBuffer commandBuffer) {
        if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
            throw std::runtime_error("Failed to record command buffer!");
        }
    }

    void copyDataToBuffer(Context* context, QueueFamilyType familyType, void* data, uint32_t size, Buffer& buffer) {
        const CopyDataToGPUBufferInfo copyDataToGPUBufferInfo {
            .data = data,
            .size = size,
            .allocator = context->allocator(),
            .buffer = buffer.get(),
            .device = context->device(),
            .queueFamilyIndex = context->familyIndex(familyType).value(),
            .queue = context->queue(familyType)
        };
        Buffer::copy(copyDataToGPUBufferInfo);
    }

    void createSSBO(VmaAllocator allocator, uint32_t size, Buffer& buffer) {
        const BufferCreateInfo bufferCreateInfo {
            .allocator = allocator,
            .size = size,
            .bufferUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .memoryUsage = VMA_MEMORY_USAGE_GPU_ONLY
        };
        buffer = Buffer(bufferCreateInfo);
    }

    void createUBO(VmaAllocator allocator, uint32_t size, Buffer& buffer) {
        const BufferCreateInfo bufferCreateInfo {
            .allocator = allocator,
            .size = size,
            .bufferUsage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .memoryUsage = VMA_MEMORY_USAGE_GPU_ONLY
        };
        buffer = Buffer(bufferCreateInfo);
    }

    VkWriteDescriptorSet getSSBODescriptorWrite(const Buffer& buffer, const uint32_t binding, std::vector<VkDescriptorBufferInfo>& infos) {
        const VkDescriptorBufferInfo bufferInfo {
            .buffer = buffer.get(),
            .offset = 0,
            .range = buffer.size()
        };
        infos.push_back(bufferInfo);
        return {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstBinding = binding,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pImageInfo = nullptr,
            .pBufferInfo = &infos.back(),
            .pTexelBufferView = nullptr
        };
    }

    VkWriteDescriptorSet getUBODescriptorWrite(const Buffer& buffer, const uint32_t binding, std::vector<VkDescriptorBufferInfo>& infos) {
        const VkDescriptorBufferInfo bufferInfo {
            .buffer = buffer.get(),
            .offset = 0,
            .range = buffer.size()
        };
        infos.push_back(bufferInfo);
        return {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstBinding = binding,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .pImageInfo = nullptr,
            .pBufferInfo = &infos.back(),
            .pTexelBufferView = nullptr
        };
    }

    VkWriteDescriptorSet getStorageImageDescriptorWrite(VkImageView view, VkImageLayout layout, uint32_t binding, std::vector<VkDescriptorImageInfo>& infos) {
        VkDescriptorImageInfo imageInfo {
            .sampler = VK_NULL_HANDLE,
            .imageView = view,
            .imageLayout = layout
        };
        infos.push_back(imageInfo);
        return {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstBinding = binding,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .pImageInfo = &infos.back(),
            .pBufferInfo = nullptr,
            .pTexelBufferView = nullptr
        };
    }

    VkWriteDescriptorSet getSamplerImageDescriptorWrite(VkSampler sampler, VkImageView view, VkImageLayout layout, uint32_t binding, std::vector<VkDescriptorImageInfo>& infos) {
        VkDescriptorImageInfo imageInfo {
            .sampler = sampler,
            .imageView = view,
            .imageLayout = layout
        };
        infos.push_back(imageInfo);
        return {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstBinding = binding,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .pImageInfo = &infos.back()
        };
    }

    VkDescriptorSetLayoutBinding getLayoutBinding(const uint32_t binding, const VkDescriptorType descriptorType, const VkShaderStageFlags stageFlags) {
        return {
            .binding = binding,
            .descriptorType = descriptorType,
            .descriptorCount = 1,
            .stageFlags = stageFlags,
            .pImmutableSamplers = nullptr
        };
    }
}
