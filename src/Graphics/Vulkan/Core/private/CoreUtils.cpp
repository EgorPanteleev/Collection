//
// Created by igor on 4/14/26.
//

#include "CoreUtils.hpp"

namespace crv::graphics::vulkan {
    std::tuple<CommandPool*, CommandBuffers*> beginCommandBuffer(const VkDevice device, const uint32_t queueFamilyIndex) {
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
}