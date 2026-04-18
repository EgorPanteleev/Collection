//
// Created by igor on 4/8/26.
//

#include "PathTracer.hpp"
#include "Message.hpp"
#include "CoreUtils.hpp"
#include "Timer.hpp"

namespace crv::graphics::vulkan {
    PathTracer::PathTracer(const PathTracerCreateInfo& info): mTriangles(info.triangles) {
        mCamera = std::make_unique<scene::FlyCamera>(info.cameraCreateInfo);
        createContext(info.windowCreateInfo);
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
        utils::FpsCounter fpsCounter;
        double deltaTime = 0;
        const Window& window = mContext.window();
        while (!window.shouldClose()) {
            glfwPollEvents();
            window.keyboardCallBack(mCamera.get(), deltaTime);
            fpsCounter.update();
            deltaTime = 1e3 / fpsCounter.fps();
            window.setTitle(std::to_string(fpsCounter.fps()).c_str());

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
            record(imageIndex);
            submit(imageIndex);
        }
        vkDeviceWaitIdle(mContext.device());
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
        constexpr VkDescriptorSetLayoutBinding binding0 {
            .binding = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .pImmutableSamplers = nullptr
        };
        constexpr VkDescriptorSetLayoutBinding binding1 {
            .binding = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .pImmutableSamplers = nullptr
        };
        constexpr VkDescriptorSetLayoutBinding binding2 {
            .binding = 2,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .pImmutableSamplers = nullptr
        };

        const std::vector bindings{binding0, binding1, binding2};
        const std::vector<VkDescriptorBindingFlags> bindingFlags{0, 0, 0};
        const DescriptorSetLayoutCreateInfo createInfo {
            .device = mContext.device(),
            .bindings = bindings,
            .bindingFlags = bindingFlags
        };
        mDescriptorSetLayout = DescriptorSetLayout(createInfo);
    }

    void PathTracer::createDescriptorPool() {
        const std::vector<VkDescriptorPoolSize> poolSizes{
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1},
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1},
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , 1},
        };

        const DescriptorPoolCreateInfo createInfo {
            .device = mContext.device(),
            .poolSizes = poolSizes,
            .maxSets = 1
        };
        mDescriptorPool = DescriptorPool(createInfo);
    }

    void PathTracer::createDescriptorSets() {
        const std::vector<uint32_t> variableCounts{0, 0, 0};
        const DescriptorSetsCreateInfo createInfo {
            .device = mContext.device(),
            .layouts = getDescriptorLayouts(),
            .pool = mDescriptorPool.get(),
            .variableCounts =variableCounts
        };
        mDescriptorSets = DescriptorSets(createInfo);
    }

    void PathTracer::createPipelineLayout() {
        VkPushConstantRange pushRange{
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .offset = 0,
            .size = sizeof(PushConstants)
        };
        const PipelineLayoutCreateInfo createInfo {
            .device = mContext.device(),
            .layouts = getDescriptorLayouts(),
            .ranges = {pushRange}
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
        const std::vector<VkPipelineShaderStageCreateInfo> stages {
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
        const ComputePipelinesCreateInfo createInfo {
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
            const ImageViewCreateInfo imageViewCreateInfo{
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
            const ImageTransitInfo imageTransitInfo {
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
        {
            AlignedCamera camera {
                .position = Vec4(mCamera->position(), 1),
                .forward = Vec4(mCamera->forward(), 1),
                .right = Vec4(mCamera->right(), 1),
                .up = Vec4(mCamera->up(), 1),
                .FOV = mCamera->FOV(),
                .aspectRatio = mCamera->aspectRatio(),
                .nearPlane = mCamera->nearPlane(),
                .farPlane = mCamera->farPlane()
            };
            const BufferCreateInfo cameraBufferCreateInfo {
                .allocator = mContext.allocator(),
                .size = sizeof(AlignedCamera),
                .bufferUsage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, //TODO
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                .memoryUsage = VMA_MEMORY_USAGE_GPU_ONLY
            };
            mCameraBuffer = Buffer(cameraBufferCreateInfo);
            CopyDataToGPUBufferInfo copyDataToGPUBufferInfo {
                .data = &camera,
                .size = sizeof(AlignedCamera),
                .allocator = mContext.allocator(),
                .buffer = mCameraBuffer.get(),
                .device = mContext.device(),
                .queueFamilyIndex = mContext.familyIndex(QueueFamilyType::COMPUTE).value(),
                .queue = mContext.queue(QueueFamilyType::COMPUTE)
            };
            Buffer::copy(copyDataToGPUBufferInfo);
        }
        VkDescriptorBufferInfo cameraBufferInfo {
            .buffer = mCameraBuffer.get(),
            .offset = 0,
            .range = mCameraBuffer.size()
        };

        {
            const uint32_t trianglesSize = sizeof(AlignedTriangle) * mTriangles.size();
            const BufferCreateInfo trianglesBufferCreateInfo {
                .allocator = mContext.allocator(),
                .size = trianglesSize,
                .bufferUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                .memoryUsage = VMA_MEMORY_USAGE_GPU_ONLY
            };
            mTrianglesBuffer = Buffer(trianglesBufferCreateInfo);
            CopyDataToGPUBufferInfo copyDataToGPUBufferInfo {
                .data = mTriangles.data(),
                .size = trianglesSize,
                .allocator = mContext.allocator(),
                .buffer = mTrianglesBuffer.get(),
                .device = mContext.device(),
                .queueFamilyIndex = mContext.familyIndex(QueueFamilyType::COMPUTE).value(),
                .queue = mContext.queue(QueueFamilyType::COMPUTE)
            };
            Buffer::copy(copyDataToGPUBufferInfo);
        }
        VkDescriptorBufferInfo trianglesBufferInfo {
            .buffer = mTrianglesBuffer.get(),
            .offset = 0,
            .range = mTrianglesBuffer.size()
        };

        const ImageCreateInfo imageCreateInfo {
            .device = mContext.device(),
            .allocator = mContext.allocator(),
            .flags = 0,
            .format = VK_FORMAT_R8G8B8A8_UNORM,
            .extent = {mSwapchain.extent().width, mSwapchain.extent().height, 1},
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .tiling = VK_IMAGE_TILING_OPTIMAL,
            .imageUsage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
            .memoryUsage = VMA_MEMORY_USAGE_AUTO
        };
        mdImage = Image(imageCreateInfo);

        auto [commandPool, commandBuffers] = beginCommandBuffer(mContext.device(), mContext.familyIndex(QueueFamilyType::COMPUTE).value());
        VkCommandBuffer commandBuffer = (*commandBuffers)[0];
        const ImageTransitInfo imageTransitInfo {
            .commandBuffer = commandBuffer,
            .image = mdImage.get(),
            .srcAccessMask = 0,
            .dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .newLayout = VK_IMAGE_LAYOUT_GENERAL,
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            .dstStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
        };
        Image::transit(imageTransitInfo);

        endCommandBuffer(commandPool, commandBuffers, mContext.queue(QueueFamilyType::COMPUTE));

        ImageViewCreateInfo imageViewCreateInfo {
            .device = mContext.device(),
            .image = mdImage.get(),
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format = imageCreateInfo.format,
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .mipLevels = 1,
            .baseMipLevel = 0,
            .baseArrayLayer = 0,
            .layerCount = 1
        };
        mdImageView = ImageView(imageViewCreateInfo);

        VkDescriptorImageInfo imageInfo {
            .sampler = VK_NULL_HANDLE,
            .imageView = mdImageView.get(),
            .imageLayout = VK_IMAGE_LAYOUT_GENERAL
        };

        VkWriteDescriptorSet writeDescriptorSet0 {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstBinding = 0,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .pImageInfo = nullptr,
            .pBufferInfo = &cameraBufferInfo,
            .pTexelBufferView = nullptr
        };

        VkWriteDescriptorSet writeDescriptorSet1 {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstBinding = 1,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pImageInfo = nullptr,
            .pBufferInfo = &trianglesBufferInfo,
            .pTexelBufferView = nullptr
        };

        VkWriteDescriptorSet writeDescriptorSet2 {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstBinding = 2,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .pImageInfo = &imageInfo,
            .pBufferInfo = nullptr,
            .pTexelBufferView = nullptr
        };
        std::vector<std::vector<VkWriteDescriptorSet>> descriptorsWrites{
            { writeDescriptorSet0, writeDescriptorSet1, writeDescriptorSet2 },
        };

        DescriptorSetsUpdateInfo updateInfo {
            .descriptorsWrites = descriptorsWrites
        };
        mDescriptorSets.update(updateInfo);
    }

    void PathTracer::record(const uint32_t imageIndex) {
        const uint32_t currentFrame = 0;//recordInfo.currentFrame;
        const auto& commandBuffer = mCommandBuffers[currentFrame];
        vkResetCommandBuffer(commandBuffer, 0);
        constexpr VkCommandBufferBeginInfo beginInfo {
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

        PushConstants pc(window().width(), window().height(), static_cast<int>(mTriangles.size()));
        vkCmdPushConstants(commandBuffer, mPipelineLayout.get(), VK_SHADER_STAGE_COMPUTE_BIT,
            0, sizeof(PushConstants), &pc);

        uint32_t groupX = (mSwapchain.extent().width + 15) / 16;
        uint32_t groupY = (mSwapchain.extent().height + 15) / 16;
        vkCmdDispatch(commandBuffer, groupX, groupY, 1);


        //render start
        const VkImageMemoryBarrier dataBarrier{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
            .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            .image = mdImage.get(),
            .subresourceRange = {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1
            }
        };

        const VkImageMemoryBarrier presentBarrier{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .srcAccessMask = 0,
            .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
            .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            .image = mImages[imageIndex],
            .subresourceRange = {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1
            }
        };

        VkImageMemoryBarrier barriers[] = {dataBarrier, presentBarrier};
        vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr,
                             0, nullptr, 2, barriers);


        VkImageBlit blit{};
        blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        blit.srcSubresource.layerCount = 1;

        blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        blit.dstSubresource.layerCount = 1;

        const auto width = static_cast<int32_t>(mSwapchain.extent().width);
        const auto height = static_cast<int32_t>(mSwapchain.extent().height);
        blit.srcOffsets[0] = {0, 0, 0};
        blit.srcOffsets[1] = {width, height, 1};

        blit.dstOffsets[0] = {0, 0, 0};
        blit.dstOffsets[1] = {width, height, 1};

        vkCmdBlitImage(
            commandBuffer,
            mdImage.get(),
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            mImages[imageIndex],
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1,
            &blit,
            VK_FILTER_NEAREST
        );

        VkImageMemoryBarrier presentBarrier1{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask = 0,
            .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            .newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
            .image = mImages[imageIndex],
            .subresourceRange = {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .levelCount = 1,
                .layerCount = 1
            }
        };

        vkCmdPipelineBarrier(
            commandBuffer,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
            0,
            0, nullptr,
            0, nullptr,
            1, &presentBarrier1
        );
        //render end


        if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
            throw std::runtime_error("Failed to record command buffer!");
        }
        vkResetFences(mContext.device(), 1, &mFence.get());
    }

    void PathTracer::submit(const uint32_t imageIndex) {
        const uint32_t currentFrame = 0;
        auto& commandBuffer = mCommandBuffers[currentFrame];
        VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT};
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