//
// Created by igor on 4/8/26.
//

#include "PathTracer.hpp"
#include "Message.hpp"
#include "CoreUtils.hpp"
#include "Timer.hpp"

namespace crv::graphics::vulkan {
    PathTracer::PathTracer(const PathTracerCreateInfo& info): mFramesInFlight(info.framesInFlight) {
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
        createBuffers(info);
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
                .imageAvailableSemaphore = mImageAvailableSemaphores[mCurrentFrame].get(),
                .fence = mFences[mCurrentFrame].get(),
                .imageIndex = &imageIndex
            };
            const VkResult result = mSwapchain.acquireNextImage(swapchainAcquireInfo);
            if (result != VK_SUCCESS) {
                throw std::runtime_error("Failed to acquire image!");
            }
            update();
            record(imageIndex);
            submit(imageIndex);
            updateCurrentFrame();
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
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .pImmutableSamplers = nullptr
        };
        constexpr VkDescriptorSetLayoutBinding binding3 {
            .binding = 3,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .pImmutableSamplers = nullptr
        };

        const std::vector bindings{binding0, binding1, binding2, binding3};
        const std::vector<VkDescriptorBindingFlags> bindingFlags{0, 0, 0, 0};
        const DescriptorSetLayoutCreateInfo createInfo {
            .device = mContext.device(),
            .bindings = bindings,
            .bindingFlags = bindingFlags
        };
        mDescriptorSetLayout = DescriptorSetLayout(createInfo);
    }

    void PathTracer::createDescriptorPool() {
        const std::vector<VkDescriptorPoolSize> poolSizes{
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, mFramesInFlight},
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, mFramesInFlight},
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, mFramesInFlight},
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE , mFramesInFlight},
        };

        const DescriptorPoolCreateInfo createInfo {
            .device = mContext.device(),
            .poolSizes = poolSizes,
            .maxSets = mFramesInFlight
        };
        mDescriptorPool = DescriptorPool(createInfo);
    }

    void PathTracer::createDescriptorSets() {
        const std::vector<uint32_t> variableCounts{0, 0, 0, 0};
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
            .bufferCount = mFramesInFlight
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
        const FenceCreateInfo fenceCreateInfo {
            .device = mContext.device()
        };
        mImageAvailableSemaphores.resize(mFramesInFlight);
        mComputeFinishedSemaphores.resize(mFramesInFlight);
        mFences.resize(mFramesInFlight);
        for (uint32_t i = 0; i < mFramesInFlight; ++i) {
            mImageAvailableSemaphores[i]  = Semaphore(semaphoreCreateInfo);
            mComputeFinishedSemaphores[i] = Semaphore(semaphoreCreateInfo);
            mFences[i] = Fence(fenceCreateInfo);
        }
    }

    void PathTracer::createBuffers(const PathTracerCreateInfo& info) {
        {
            const uint32_t trianglesSize = sizeof(AlignedTriangle) * info.triangles.size();
            const BufferCreateInfo triangleBufferCreateInfo {
                .allocator = mContext.allocator(),
                .size = trianglesSize,
                .bufferUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                .memoryUsage = VMA_MEMORY_USAGE_GPU_ONLY
            };
            mTriangleBuffer = Buffer(triangleBufferCreateInfo);
            const CopyDataToGPUBufferInfo copyDataToGPUBufferInfo {
                .data = const_cast<AlignedTriangle*>(info.triangles.data()),
                .size = trianglesSize,
                .allocator = mContext.allocator(),
                .buffer = mTriangleBuffer.get(),
                .device = mContext.device(),
                .queueFamilyIndex = mContext.familyIndex(QueueFamilyType::COMPUTE).value(),
                .queue = mContext.queue(QueueFamilyType::COMPUTE)
            };
            Buffer::copy(copyDataToGPUBufferInfo);
        }
        {
            const uint32_t nodesSize = sizeof(AlignedNode) * info.nodes.size();
            const BufferCreateInfo nodeBufferCreateInfo {
                .allocator = mContext.allocator(),
                .size = nodesSize,
                .bufferUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                .memoryUsage = VMA_MEMORY_USAGE_GPU_ONLY
            };
            mNodeBuffer = Buffer(nodeBufferCreateInfo);
            const CopyDataToGPUBufferInfo copyDataToGPUBufferInfo {
                .data = const_cast<AlignedNode*>(info.nodes.data()),
                .size = nodesSize,
                .allocator = mContext.allocator(),
                .buffer = mNodeBuffer.get(),
                .device = mContext.device(),
                .queueFamilyIndex = mContext.familyIndex(QueueFamilyType::COMPUTE).value(),
                .queue = mContext.queue(QueueFamilyType::COMPUTE)
            };
            Buffer::copy(copyDataToGPUBufferInfo);
        }
        {
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

            const ImageViewCreateInfo imageViewCreateInfo {
                .device = mContext.device(),
                .image = mdImage.get(),
                .viewType = VK_IMAGE_VIEW_TYPE_2D,
                .format = VK_FORMAT_R8G8B8A8_UNORM,
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .mipLevels = 1,
                .baseMipLevel = 0,
                .baseArrayLayer = 0,
                .layerCount = 1
            };
            mdImageView = ImageView(imageViewCreateInfo);
        }
        mCameraBuffers.resize(mFramesInFlight);
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
                .bufferUsage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                .memoryUsage = VMA_MEMORY_USAGE_GPU_ONLY
            };
            mCameraBuffers[mCurrentFrame] = Buffer(cameraBufferCreateInfo);
            CopyDataToGPUBufferInfo copyDataToGPUBufferInfo {
                .data = &camera,
                .size = sizeof(AlignedCamera),
                .allocator = mContext.allocator(),
                .buffer = mCameraBuffers[mCurrentFrame].get(),
                .device = mContext.device(),
                .queueFamilyIndex = mContext.familyIndex(QueueFamilyType::COMPUTE).value(),
                .queue = mContext.queue(QueueFamilyType::COMPUTE)
            };
            Buffer::copy(copyDataToGPUBufferInfo);
        }
        VkDescriptorBufferInfo cameraBufferInfo {
            .buffer = mCameraBuffers[mCurrentFrame].get(),
            .offset = 0,
            .range = mCameraBuffers[mCurrentFrame].size()
        };
        VkDescriptorBufferInfo trianglesBufferInfo {
            .buffer = mTriangleBuffer.get(),
            .offset = 0,
            .range = mTriangleBuffer.size()
        };
        VkDescriptorBufferInfo nodeBufferInfo {
            .buffer = mNodeBuffer.get(),
            .offset = 0,
            .range = mNodeBuffer.size()
        };

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
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pImageInfo = nullptr,
            .pBufferInfo = &nodeBufferInfo,
            .pTexelBufferView = nullptr
        };

        VkWriteDescriptorSet writeDescriptorSet3 {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstBinding = 3,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .pImageInfo = &imageInfo,
            .pBufferInfo = nullptr,
            .pTexelBufferView = nullptr
        };
        std::vector descriptorWrites{
            writeDescriptorSet0, writeDescriptorSet1,
            writeDescriptorSet2, writeDescriptorSet3
        };
        std::vector<std::vector<VkWriteDescriptorSet>> descriptorsWrites;
        for (uint32_t i = 0; i < mFramesInFlight; ++i) {
            descriptorsWrites.push_back(descriptorWrites);
        }

        DescriptorSetsUpdateInfo updateInfo {
            .descriptorsWrites = descriptorsWrites
        };
        mDescriptorSets.update(updateInfo);
    }

    void PathTracer::record(const uint32_t imageIndex) {
        const auto& commandBuffer = mCommandBuffers[mCurrentFrame];
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
                        0, 1, &mDescriptorSets[mCurrentFrame], 0, nullptr);

        const uint32_t width  = mSwapchain.extent().width;
        const uint32_t height = mSwapchain.extent().height;
        const PushConstants pc(width, height);
        vkCmdPushConstants(commandBuffer, mPipelineLayout.get(), VK_SHADER_STAGE_COMPUTE_BIT,
            0, sizeof(PushConstants), &pc);

        const uint32_t groupX = (width  + 7) / 8;
        const uint32_t groupY = (height + 7) / 8;
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

        blit.srcOffsets[0] = {0, 0, 0};
        blit.srcOffsets[1] = {static_cast<int32_t>(width), static_cast<int32_t>(height), 1};

        blit.dstOffsets[0] = {0, 0, 0};
        blit.dstOffsets[1] = {static_cast<int32_t>(width), static_cast<int32_t>(height), 1};

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
        vkResetFences(mContext.device(), 1, &mFences[mCurrentFrame].get());
    }

    void PathTracer::submit(const uint32_t imageIndex) {
        auto& commandBuffer = mCommandBuffers[mCurrentFrame];
        VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT};
        const VkSubmitInfo submitInfo{
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &mImageAvailableSemaphores[mCurrentFrame].get(),
            .pWaitDstStageMask = waitStages,
            .commandBufferCount = 1,
            .pCommandBuffers = &commandBuffer,
            .signalSemaphoreCount = 1,
            .pSignalSemaphores = &mComputeFinishedSemaphores[mCurrentFrame].get()
        };

        VkQueue queue = mContext.queue(QueueFamilyType::COMPUTE);
        if (vkQueueSubmit(queue, 1, &submitInfo, mFences[mCurrentFrame].get()) != VK_SUCCESS) {
            throw std::runtime_error("Failed to submit draw command buffer!");
        }
        const VkPresentInfoKHR presentInfo {
            .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &mComputeFinishedSemaphores[mCurrentFrame].get(),
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
        for (uint32_t i = 0; i < mFramesInFlight; ++i) {
            layouts.push_back(mDescriptorSetLayout.get());
        }
        return layouts;
    }

    std::vector<VkPipelineLayout> PathTracer::getPipelineLayouts() const {
        std::vector<VkPipelineLayout> layouts;
        layouts.push_back(mPipelineLayout.get());
        return layouts;
    }
}