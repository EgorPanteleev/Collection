//
// Created by igor on 4/8/26.
//

#include "PathTracer.hpp"
#include "Message.hpp"
#include "Utils.hpp"

namespace crv::graphics::vulkan {
    PathTracer::PathTracer(const WindowCreateInfo& windowCreateInfo) {
        createContext(windowCreateInfo);
        createDescriptorSetLayout();
        createDescriptorPool();
        createDescriptorSets();
        createPipelineLayout();
        createShaders();
        createComputePipelines();
        createCommandPool();
        createCommandBuffers();
        createSwapChain();
        createImages();
        createSyncObjects();
    }

    void PathTracer::run() {
        //acquire -> update -> record -> reset fences -> submit
        while (true) {
            uint32_t imageIndex;
            SwapchainAcquireInfo swapchainAcquireInfo {
                .imageAvailableSemaphore = mImageAvailableSemaphore.get(),
                .fence = mFence.get(),
                .imageIndex = &imageIndex
            };
            const VkResult result = mSwapchain.acquireNextImage(swapchainAcquireInfo);
            if (result != VK_SUCCESS) {
                throw std::runtime_error("Failed to acquire image!");
            }
            update();
            record();
            submit(imageIndex);

            auto data = new float[3];
            VkDeviceSize size = sizeof(float) * 3;
            CopyGPUBufferToDataInfo copyInfo {
                .data = data,
                .size = size,
                .allocator = mContext.allocator(),
                .buffer = mBuffer2.get(),
                .device = mContext.device(),
                .queueFamilyIndex = mContext.familyIndex(QueueFamilyType::COMPUTE).value(),
                .queue = mContext.queue(QueueFamilyType::COMPUTE)
            };
            Buffer::copy(copyInfo);

            for (int i = 0; i < 3; ++i) {
                MESSAGE << data[i];
            }
        }
    }

    void PathTracer::createContext(const WindowCreateInfo& windowCreateInfo) {
        const ContextCreateInfo createInfo {
            .windowCreateInfo = windowCreateInfo,
            .validationLayers = { "VK_LAYER_KHRONOS_validation" },
            .deviceExtensions = { VK_KHR_SWAPCHAIN_EXTENSION_NAME,
                                     VK_KHR_MAINTENANCE_1_EXTENSION_NAME },
            .enableValidationLayers = mDebug
        };
        mContext = Context(createInfo);
    }

    void PathTracer::createDescriptorSetLayout() {
        const VkDescriptorSetLayoutBinding binding0 {
            .binding = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .pImmutableSamplers = nullptr
        };
        const VkDescriptorSetLayoutBinding binding1 {
            .binding = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .pImmutableSamplers = nullptr
        };

        std::vector bindings{binding0,
                             binding1};
        std::vector<VkDescriptorBindingFlags> bindingFlags{0,
                                                           0};
        const DescriptorSetLayoutCreateInfo createInfo {
            .device = mContext.device(),
            .bindings = bindings,
            .bindingFlags = bindingFlags
        };
        mDescriptorSetLayout = DescriptorSetLayout(createInfo);
    }

    void PathTracer::createDescriptorPool() {
        std::vector<VkDescriptorPoolSize> poolSizes {
                {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1},
                {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1},
        };

        const DescriptorPoolCreateInfo createInfo {
            .device = mContext.device(),
            .poolSizes = poolSizes,
            .maxSets = 1
        };
        mDescriptorPool = DescriptorPool(createInfo);
    }

    void PathTracer::createDescriptorSets() {
        std::vector<uint32_t> variableCounts{0, 0};
        DescriptorSetsCreateInfo createInfo {
            .device = mContext.device(),
            .layouts = getDescriptorLayouts(),
            .pool = mDescriptorPool.get(),
            .variableCounts =variableCounts
        };
        mDescriptorSets = DescriptorSets(createInfo);
    }

    void PathTracer::createPipelineLayout() {
        PipelineLayoutCreateInfo createInfo {
            .device = mContext.device(),
            .layouts = getDescriptorLayouts()
        };
        mPipelineLayout = PipelineLayout(createInfo);
    }

    void PathTracer::createShaders() {
        const ShaderModuleCreateInfo createInfo {
            .device = mContext.device(),
            .fileName = COMPILED_SHADERS_DIR"/shader1.comp.spv"
        };
        mShader = ShaderModule(createInfo);
    }

    void PathTracer::createComputePipelines() {
        std::vector<VkPipelineShaderStageCreateInfo> stages {
            {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                .module = mShader.get(),
                .pName = "main",
                .pSpecializationInfo = nullptr
            },
        };
        ComputePipelinesCreateInfo createInfo {
            .device = mContext.device(),
            .stages = stages,
            .layouts = getPipelineLayouts(),
        };
        mComputePipelines = ComputePipelines(createInfo);
    }

    void PathTracer::createCommandPool() {
        const CommandPoolCreateInfo createInfo {
            .device = mContext.device(),
            .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            .queueFamilyIndex = mContext.familyIndex(QueueFamilyType::COMPUTE).value()
        };
        mCommandPool = CommandPool(createInfo);
    }

    void PathTracer::createCommandBuffers() {
        const CommandBuffersCreateInfo createInfo {
            .device = mContext.device(),
            .commandPool = mCommandPool.get(),
            .bufferCount = 1
        };
        mCommandBuffers = CommandBuffers(createInfo);
    }

    void PathTracer::createSwapChain() {
        int width, height;
        mContext.window().getFrameBufferSize(width, height);
        const SwapchainCreateInfo info{
            .device = mContext.device(),
            .physicalDevice = mContext.physicalDevice(),
            .surface = mContext.surface(),
            .windowWidth = static_cast<uint32_t>(width),
            .windowHeight = static_cast<uint32_t>(height),
            .familyIndices = mContext.familyIndices()
        };
        mSwapchain = Swapchain(info);
    }

    void PathTracer::createImages() {
        auto [capabilities, formats, presentModes] = Swapchain::getSupport(mContext.physicalDevice(), mContext.surface());
        uint32_t imageCount = Swapchain::getImageCount(capabilities);
        vkGetSwapchainImagesKHR(mContext.device(), mSwapchain.get(), &imageCount, nullptr);
        mImages.resize(imageCount);
        vkGetSwapchainImagesKHR(mContext.device(), mSwapchain.get(), &imageCount, mImages.data());

        auto [commandPool, commandBuffers] = beginCommandBuffer(mContext.device(), mContext.familyIndex(QueueFamilyType::COMPUTE).value());
        VkCommandBuffer commandBuffer = (*commandBuffers)[0];
        for (uint32_t i = 0; i < imageCount; ++i) {
            ImageViewCreateInfo imageViewCreateInfo{
                .device = mContext.device(),
                .image = mImages[i],
                .viewType = VK_IMAGE_VIEW_TYPE_2D,
                .format = mSwapchain.format(),
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .mipLevels = 1,
                .baseMipLevel = 0,
                .baseArrayLayer = 0,
                .layerCount = 1
            };
            mImageViews.emplace_back(imageViewCreateInfo);
            ImageTransitInfo imageTransitInfo {
                .commandBuffer = commandBuffer,
                .image = mImages[i],
                .srcAccessMask = VK_ACCESS_NONE,
                .dstAccessMask = VK_ACCESS_NONE,
                .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                .newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
                .srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                .dstStage = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT
            };
            Image::transit(imageTransitInfo);
        }
        endCommandBuffer(commandPool, commandBuffers, mContext.queue(QueueFamilyType::COMPUTE));
    }

    void PathTracer::createSyncObjects() {
        const SemaphoreCreateInfo semaphoreCreateInfo {
            .device = mContext.device()
        };
        mImageAvailableSemaphore = Semaphore(semaphoreCreateInfo);
        mComputeFinishedSemaphore = Semaphore(semaphoreCreateInfo);
        const FenceCreateInfo fenceCreateInfo {
            .device = mContext.device()
        };
        mFence = Fence(fenceCreateInfo);
    }

    void PathTracer::update() {
        std::vector<float> data1;
        data1.push_back(1);
        data1.push_back(2);
        data1.push_back(3);
        auto data2 = data1;
        BufferCreateInfo bufferCreateInfo {
            .allocator = mContext.allocator(),
            .size = sizeof(float) * data1.size(),
            .bufferUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .memoryUsage = VMA_MEMORY_USAGE_GPU_ONLY
        };
        mBuffer1 = Buffer(bufferCreateInfo);
        CopyDataToGPUBufferInfo copyDataToGPUBufferInfo {
            .data = data1.data(),
            .size = sizeof(float) * data1.size(),
            .allocator = mContext.allocator(),
            .buffer = mBuffer1.get(),
            .device = mContext.device(),
            .queueFamilyIndex = mContext.familyIndex(QueueFamilyType::COMPUTE).value(),
            .queue = mContext.queue(QueueFamilyType::COMPUTE)
        };
        Buffer::copy(copyDataToGPUBufferInfo);
        bufferCreateInfo.size = sizeof(float) * data2.size();
        mBuffer2 = Buffer(bufferCreateInfo);
        copyDataToGPUBufferInfo.data = data2.data();
        copyDataToGPUBufferInfo.size = sizeof(float) * data2.size();
        copyDataToGPUBufferInfo.buffer = mBuffer2.get();
        Buffer::copy(copyDataToGPUBufferInfo);

        VkDescriptorBufferInfo buffer1Info {
            .buffer = mBuffer1.get(),
            .offset = 0,
            .range = mBuffer1.size()
        };

        VkDescriptorBufferInfo buffer2Info {
            .buffer = mBuffer2.get(),
            .offset = 0,
            .range = mBuffer2.size()
        };

        std::vector<std::vector<VkWriteDescriptorSet>> descriptorsWrites{
            { //1
                {
                    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .dstBinding = 0,
                    .dstArrayElement = 0,
                    .descriptorCount = 1,
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    .pImageInfo = nullptr,
                    .pBufferInfo = &buffer1Info,
                    .pTexelBufferView = nullptr
                },
                {
                    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .dstBinding = 1,
                    .dstArrayElement = 0,
                    .descriptorCount = 1,
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    .pImageInfo = nullptr,
                    .pBufferInfo = &buffer2Info,
                    .pTexelBufferView = nullptr
                },
            },
        };

        DescriptorSetsUpdateInfo updateInfo {
            .descriptorsWrites = descriptorsWrites
        };
        mDescriptorSets.update(updateInfo);
    }

    void PathTracer::record() {
        const uint32_t currentFrame = 0;//recordInfo.currentFrame;
        const auto& commandBuffer = mCommandBuffers[currentFrame];
        vkResetCommandBuffer(commandBuffer, 0);
        const VkCommandBufferBeginInfo beginInfo {
                .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                .flags = 0,
                .pInheritanceInfo = nullptr
        };
        if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
            throw std::runtime_error("Failed to begin recording command buffer!");
        }
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, mComputePipelines[0]);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, mPipelineLayout.get(),
                        0, 1, &mDescriptorSets[currentFrame], 0, nullptr);
        vkCmdDispatch(commandBuffer, 120, 68, 1);

        if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
            throw std::runtime_error("Failed to record command buffer!");
        }
        vkResetFences(mContext.device(), 1, &mFence.get());
    }

    void PathTracer::submit(const uint32_t imageIndex) {
        const uint32_t currentFrame = 0;
        auto& commandBuffer = mCommandBuffers[currentFrame];
        VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
        const VkSubmitInfo submitInfo{
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &mImageAvailableSemaphore.get(),
            .pWaitDstStageMask = waitStages,
            .commandBufferCount = 1,
            .pCommandBuffers = &commandBuffer,
            .signalSemaphoreCount = 1,
            .pSignalSemaphores = &mComputeFinishedSemaphore.get()
        };

        VkQueue queue = mContext.queue(QueueFamilyType::COMPUTE);
        if (vkQueueSubmit(queue, 1, &submitInfo, mFence.get()) != VK_SUCCESS) {
            throw std::runtime_error("Failed to submit draw command buffer!");
        }

        const VkPresentInfoKHR presentInfo {
            .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &mComputeFinishedSemaphore.get(),
            .swapchainCount = 1,
            .pSwapchains = &mSwapchain.get(),
            .pImageIndices = &imageIndex,
            .pResults = nullptr
        };
        const VkResult result = vkQueuePresentKHR(mContext.queue(QueueFamilyType::PRESENT), &presentInfo);
        if (result != VK_SUCCESS) {
            throw std::runtime_error("Failed to present image!");
        }
    }

    std::vector<VkDescriptorSetLayout> PathTracer::getDescriptorLayouts() const {
        std::vector<VkDescriptorSetLayout> layouts;
        layouts.push_back(mDescriptorSetLayout.get());
        return layouts;
    }

    std::vector<VkPipelineLayout> PathTracer::getPipelineLayouts() const {
        std::vector<VkPipelineLayout> layouts;
        layouts.push_back(mPipelineLayout.get());
        return layouts;
    }
}

/* Plan
 * 1) Creating context
 *   1.1) window, surface
 *   1.2) instance
 *   1.3) devices
 *   1.4) queues
 *   1.5) allocator
 *
 * 2) Buffers - resources
 *   2.1) BVH
 *   2.2) Materials
 *   2.3) Lights
 *
 * 3) Bind resources
 *   3.1) Descriptor Set Layout
 *   3.2) Descriptor Set
 *
 * 4) Bind descriptor sets
 *   4.1) Write shaders, compile it
 *   4.2) Compute Pipeline Layout
 *   4.3) Compute Pipeline
 *
 * 5) Command buffers, recording
 *   5.1) Record command buffer
 *   5.2) Submit command buffer
 *
 * 6) Present the results
*/