//
// Created by igor on 4/14/26.
//

#ifndef COLLECTION_COREUTILS_HPP
#define COLLECTION_COREUTILS_HPP

#include "CommandPool.hpp"
#include "CommandBuffers.hpp"
#include "Context.hpp"
#include "Buffer.hpp"
#include "ImageView.hpp"
#include "Texture.hpp"

#include <glm/glm.hpp>

namespace crv::graphics::vulkan {
    struct CommandBufferData {
        CommandPool*    commandPool    = nullptr;
        CommandBuffers* commandBuffers = nullptr;
    };

    std::tuple<VkCommandBuffer, CommandBufferData> beginCommandBuffer(Context* context, QueueFamilyType type);
    std::tuple<VkCommandBuffer, CommandBufferData> beginCommandBuffer(VkDevice device, uint32_t queueFamilyIndex);
    void endCommandBuffer(const CommandBufferData& data, VkQueue queue);
    void beginCommandBuffer(VkCommandBuffer commandBuffer);
    void endCommandBuffer(VkCommandBuffer commandBuffer);
    void copyDataToBuffer(Context* context, QueueFamilyType familyType, const void* data, uint32_t size, Buffer& buffer);
    void createSSBO(VmaAllocator allocator, uint32_t size, Buffer& buffer);
    void createUBO(VmaAllocator allocator, uint32_t size, Buffer& buffer);
    VkWriteDescriptorSet getSSBODescriptorWrite(const Buffer& buffer, uint32_t binding, std::vector<VkDescriptorBufferInfo>& infos);
    VkWriteDescriptorSet getUBODescriptorWrite(const Buffer& buffer, uint32_t binding, std::vector<VkDescriptorBufferInfo>& infos);
    VkWriteDescriptorSet getStorageImageDescriptorWrite(VkImageView view, VkImageLayout layout, uint32_t binding, std::vector<VkDescriptorImageInfo>& infos);
    VkWriteDescriptorSet getSamplerImageDescriptorWrite(VkSampler sampler, VkImageView view, VkImageLayout layout, uint32_t binding, std::vector<VkDescriptorImageInfo>& infos);
    VkDescriptorSetLayoutBinding getLayoutBinding(uint32_t binding, VkDescriptorType descriptorType, VkShaderStageFlags stageFlags);
    VkViewport getDefaultViewport(VkExtent2D extent);
    VkRect2D getDefaultScissor(VkExtent2D extent);
    VkTransformMatrixKHR toVkTransform(const glm::mat4& mat);
    Texture toTexture(Context* context, const cm::Texture& texture);
}
#endif //COLLECTION_COREUTILS_HPP