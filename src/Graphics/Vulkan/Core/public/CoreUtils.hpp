//
// Created by igor on 4/14/26.
//

#ifndef COLLECTION_COREUTILS_HPP
#define COLLECTION_COREUTILS_HPP

#include "CommandPool.hpp"
#include "CommandBuffers.hpp"

namespace crv::graphics::vulkan {
    std::tuple<CommandPool*, CommandBuffers*> beginCommandBuffer(VkDevice device, uint32_t queueFamilyIndex);
    void endCommandBuffer(CommandPool* commandPool, CommandBuffers* commandBuffers, VkQueue queue);
    void beginCommandBuffer(VkCommandBuffer commandBuffer);
    void endCommandBuffer(VkCommandBuffer commandBuffer);
}

#endif //COLLECTION_COREUTILS_HPP