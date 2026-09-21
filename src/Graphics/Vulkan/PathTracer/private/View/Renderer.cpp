//
// Created by igor on 6/7/26.
//

#include "View/Renderer.hpp"
#include "CoreUtils.hpp"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <chrono>
#include <filesystem>

namespace fs = std::filesystem;

namespace crv::graphics::vulkan {
    Renderer::Renderer(const RendererCreateInfo& info) : mModel(info.model), mCommands(info.commands) {
        createContext(info.windowCreateInfo);
        createSwapChain();
        createSwapChainImages();
        createBuffers();
        createImages();
        createSyncObjects();
        createCommandBuffers();
        createResourceManager();
        createRayTracerPass();
        createRasterizerPass();
        createPostprocessPass();
    }

    void Renderer::createContext(const WindowCreateInfo& windowCreateInfo) {
        const ContextCreateInfo createInfo {
            .windowCreateInfo = windowCreateInfo,
            .validationLayers = { "VK_LAYER_KHRONOS_validation" },
            .deviceExtensions = { VK_KHR_SWAPCHAIN_EXTENSION_NAME,
                                     VK_KHR_MAINTENANCE_1_EXTENSION_NAME,
                                     VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
                                     VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
                                     VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
                                     VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
                                     VK_KHR_SPIRV_1_4_EXTENSION_NAME,
                                     VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME,
                                     VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME},
            .enableValidationLayers = mDebug,
            .enableRT               = true
        };
        mContext = Context(createInfo);
    }

    void Renderer::createSwapChain() {
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

    void Renderer::createBuffers() {
        const BufferCreateInfo readbackInfo {
            .allocator = mContext.allocator(),
            .size = sizeof(uint32_t),
            .bufferUsage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .memoryUsage = VMA_MEMORY_USAGE_CPU_ONLY
        };
        mReadbackBuffer = Buffer(readbackInfo);
    }

    void Renderer::createImages() {
        ImageCreateInfo imageCreateInfo {
            .device = mContext.device(),
            .allocator = mContext.allocator(),
            .flags = 0,
            .format = VK_FORMAT_R16G16B16A16_SFLOAT,
            .extent = {mSwapchain.extent().width, mSwapchain.extent().height, 1},
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .tiling = VK_IMAGE_TILING_OPTIMAL,
            .imageUsage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
            .memoryUsage = VMA_MEMORY_USAGE_AUTO
        };
        imageCreateInfo.format = VK_FORMAT_R32G32B32A32_SFLOAT;
        mTracerImage = Image(imageCreateInfo);
        imageCreateInfo.format = VK_FORMAT_R16G16B16A16_SFLOAT;
        mFinalImage  = Image(imageCreateInfo);

        imageCreateInfo.format = VK_FORMAT_R32_UINT;
        mTracerInstanceImage = Image(imageCreateInfo);
        imageCreateInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
        mRasterInstanceImage = Image(imageCreateInfo);

        ImageViewCreateInfo imageViewCreateInfo {
            .device = mContext.device(),
            .image = mTracerImage.get(),
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format = VK_FORMAT_R32G32B32A32_SFLOAT,
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
        };
        mTracerView = ImageView(imageViewCreateInfo);
        imageViewCreateInfo.image = mFinalImage.get();
        imageViewCreateInfo.format = VK_FORMAT_R16G16B16A16_SFLOAT;
        mFinalView = ImageView(imageViewCreateInfo);

        imageViewCreateInfo.image = mTracerInstanceImage.get();
        imageViewCreateInfo.format = VK_FORMAT_R32_UINT;
        mTracerInstanceView = ImageView(imageViewCreateInfo);
        imageViewCreateInfo.image = mRasterInstanceImage.get();
        mRasterInstanceView = ImageView(imageViewCreateInfo);

        auto [commandBuffer, cmdData] = beginCommandBuffer(mContext.device(),
                                            mContext.familyIndex(QueueFamilyType::GRAPHICS).value());
        const ImageTransitInfo2 tracerTransitInfo {
            .commandBuffer = commandBuffer,
            .image = mTracerImage.get(),
            .srcAccessMask = VK_ACCESS_2_NONE,
            .dstAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
            .srcStage = VK_PIPELINE_STAGE_2_NONE,
            .dstStage = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
            .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .newLayout = VK_IMAGE_LAYOUT_GENERAL,
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
        };
        ImageTransitInfo2 tracerInstanceTransitInfo = tracerTransitInfo;
        tracerInstanceTransitInfo.image = mTracerInstanceImage.get();

        const ImageTransitInfo2 rasterTransitInfo {
            .commandBuffer = commandBuffer,
            .image = mRasterInstanceImage.get(),
            .srcAccessMask = VK_ACCESS_2_NONE,
            .dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
            .srcStage = VK_PIPELINE_STAGE_2_NONE,
            .dstStage = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
        };

        const ImageTransitInfo2 finalTransitInfo {
            .commandBuffer = commandBuffer,
            .image = mFinalImage.get(),
            .srcAccessMask = VK_ACCESS_2_NONE,
            .dstAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
            .srcStage = VK_PIPELINE_STAGE_2_NONE,
            .dstStage = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .newLayout = VK_IMAGE_LAYOUT_GENERAL,
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
        };
        Image::transit({tracerTransitInfo, tracerInstanceTransitInfo, rasterTransitInfo, finalTransitInfo});
        endCommandBuffer(cmdData, mContext.queue(QueueFamilyType::GRAPHICS));
    }

    void Renderer::createSwapChainImages() {
        auto [capabilities, formats, presentModes] = Swapchain::getSupport(mContext.physicalDevice(), mContext.surface());
        uint32_t imageCount = Swapchain::getImageCount(capabilities);
        vkGetSwapchainImagesKHR(mContext.device(), mSwapchain.get(), &imageCount, nullptr);
        mSwapchainImages.resize(imageCount);
        vkGetSwapchainImagesKHR(mContext.device(), mSwapchain.get(), &imageCount, mSwapchainImages.data());

        mSwapchainImageViews.reserve(imageCount);
        for (uint32_t i = 0; i < imageCount; ++i) {
            const ImageViewCreateInfo imageViewCreateInfo{
                .device = mContext.device(),
                .image = mSwapchainImages[i],
                .viewType = VK_IMAGE_VIEW_TYPE_2D,
                .format = mSwapchain.format(),
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            };
            mSwapchainImageViews.emplace_back(imageViewCreateInfo);
        }
    }

    void Renderer::createSyncObjects() {
        auto [capabilities, formats, presentModes] = Swapchain::getSupport(mContext.physicalDevice(), mContext.surface());
        uint32_t imageCount = Swapchain::getImageCount(capabilities);
        const SemaphoreCreateInfo semaphoreCreateInfo {
            .device = mContext.device()
        };
        const FenceCreateInfo fenceCreateInfo {
            .device = mContext.device()
        };
        mFences.resize(mFramesInFlight);
        mImageAvailableSemaphores.resize(mFramesInFlight);
        mTracerFinishedSemaphores.resize(mFramesInFlight);
        mRasterFinishedSemaphores.resize(mFramesInFlight);
        for (uint32_t i = 0; i < mFramesInFlight; ++i) {
            mFences[i] = Fence(fenceCreateInfo);
            mImageAvailableSemaphores[i] = Semaphore(semaphoreCreateInfo);
            mTracerFinishedSemaphores[i] = Semaphore(semaphoreCreateInfo);
            mRasterFinishedSemaphores[i] = Semaphore(semaphoreCreateInfo);
        }
        mPostprocessFinishedSemaphores.resize(imageCount);
        for (uint32_t i = 0; i < imageCount; ++i) {
            mPostprocessFinishedSemaphores[i] = Semaphore(semaphoreCreateInfo);
        }
    }

    void Renderer::createCommandBuffers() {
        const CommandPoolCreateInfo poolCreateInfo {
            .device = mContext.device(),
            .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            .queueFamilyIndex = mContext.familyIndex(QueueFamilyType::GRAPHICS).value()
        };
        mTracerCommandPool = CommandPool(poolCreateInfo);
        mRasterCommandPool = CommandPool(poolCreateInfo);
        mPostprocessCommandPool = CommandPool(poolCreateInfo);

        CommandBuffersCreateInfo bufferCreateInfo {
            .device = mContext.device(),
            .bufferCount = mFramesInFlight
        };
        bufferCreateInfo.commandPool = mTracerCommandPool.get();
        mTracerCommandBuffers = CommandBuffers(bufferCreateInfo);
        bufferCreateInfo.commandPool = mRasterCommandPool.get();
        mRasterCommandBuffers = CommandBuffers(bufferCreateInfo);
        bufferCreateInfo.commandPool = mPostprocessCommandPool.get();
        mPostprocessCommandBuffers = CommandBuffers(bufferCreateInfo);
    }

    void Renderer::createResourceManager() {
        const ResourceManagerCreateInfo createInfo {
            .context = &mContext,
            .scene   = &mModel->scene()
        };
        mResourceManager = ResourceManager(createInfo);
    }

    void Renderer::createRayTracerPass() {
        const RayTracerPassCreateInfo createInfo {
            .context = &mContext,
            .tlas = &mResourceManager.tlas(),
            .BLASBuffer = &mResourceManager.blasBuffer(),
            .instanceBuffer = &mResourceManager.instanceBuffer(),
            .emissiveInstanceBuffer = &mResourceManager.emissiveInstanceBuffer(),
            .materialBuffer = &mResourceManager.materialBuffer(),
            .textures = &mResourceManager.textures(),
            .outView = &mTracerView,
            .outInstanceIdView = &mTracerInstanceView,
            .framesInFlight = mFramesInFlight
        };
        mRayTracerPass = RayTracerPass(createInfo);
    }

    void Renderer::createRasterizerPass() {
        const RasterizerPassCreateInfo createInfo {
            .context = &mContext,
            .outView = &mRasterInstanceView,
            .outFormat = VK_FORMAT_R32_UINT,
            .extent = mSwapchain.extent(),
            .framesInFlight = mFramesInFlight
        };
        mRasterizerPass = RasterizerPass(createInfo);
    }

    void Renderer::createPostprocessPass() {
        const PostprocessPassCreateInfo createInfo {
            .context = &mContext,
            .tracerView = &mTracerView,
            .instanceView = &mRasterInstanceView,
            .outputView = &mFinalView,
            .framesInFlight = mFramesInFlight
        };
        mPostprocessPass = PostprocessPass(createInfo);
    }

    void Renderer::initUI() {
        const AppUICreateInfo createInfo {
            .context = &mContext,
            .swapchain = &mSwapchain,
            .renderSettings = &mModel->settings(),
            .commands = mCommands,
            .scene = &mModel->scene()
        };
        mUI = AppUI(createInfo);
    }

    void Renderer::recordTracer() {
        VkCommandBuffer commandBuffer = mTracerCommandBuffers[mCurrentFrame];
        vkResetCommandBuffer(commandBuffer, 0);
        beginCommandBuffer(commandBuffer);
        const RayTracerPassRecordInfo recordInfo {
            .commandBuffer = commandBuffer,
            .constants = {
                .frameCount = mFrameCount,
                .spp = static_cast<uint32_t>(mModel->settings().spp),
                .minDepth = static_cast<uint32_t>(mModel->settings().minDepth),
                .maxDepth = static_cast<uint32_t>(mModel->settings().maxDepth),
                .displayMode = static_cast<uint32_t>(mModel->settings().displayMode),
                .nee = mModel->settings().nee ? 1u : 0u,
                .emissiveCount = static_cast<uint32_t>(mModel->scene().emissiveIndices().size()),
                .skyboxIndex = mModel->scene().skyboxIndex(),
                .envIntegral = mResourceManager.envIntegral(),
                .envNee = mModel->settings().envNee ? 1u : 0u,
                .aperture = mModel->settings().aperture,
                .focusDistance = mModel->settings().focusDistance,
                .envMarginalCdfAddr = mResourceManager.envMarginalCdfAddr(),
                .envCondCdfAddr = mResourceManager.envCondCdfAddr(),
                .envCondFuncAddr = mResourceManager.envCondFuncAddr(),
                .skyColor = mModel->scene().skyColor(),
                .emissivePowerInv = mResourceManager.emissivePowerInv(),
                .envRotation = mModel->scene().envRotation()
            },
            .width = (mSwapchain.extent().width + mEffectiveScale - 1) / mEffectiveScale,
            .height = (mSwapchain.extent().height + mEffectiveScale - 1) / mEffectiveScale,
            .currentFrame = mCurrentFrame
        };
        mRayTracerPass.record(recordInfo);
        recordPixelRead(commandBuffer);
        endCommandBuffer(commandBuffer);
    }

    void Renderer::recordRaster() {
        VkCommandBuffer commandBuffer = mRasterCommandBuffers[mCurrentFrame];
        vkResetCommandBuffer(commandBuffer, 0);
        beginCommandBuffer(commandBuffer);
        std::vector<RasterizerDraw> draws;
        const auto& instances = mModel->scene().instances();
        const std::vector<bool> outlined =
            mModel->scene().subtreeMask(mModel->selection().selectedInstances);
        uint32_t outlineId = 1;
        for (uint32_t i = 0; i < instances.size(); ++i) {
            if (!outlined[i]) continue;
            const InstanceData& instance = instances[i];
            if (instance.isGroup()) continue;
            BLASData& blasData = mResourceManager.blasDatas()[instance.meshIndex];
            draws.push_back({
                .vertexBuffer = &blasData.vertexBuffer,
                .indexBuffer = &blasData.indexBuffer,
                .indexCount = instance.indexCount,
                .model = instance.world,
                .id = outlineId++
            });
        }
        const RasterizerPassRecordInfo recordInfo {
            .commandBuffer = commandBuffer,
            .draws = std::move(draws),
            .extent = mSwapchain.extent(),
            .currentFrame = mCurrentFrame
        };
        mRasterizerPass.record(recordInfo);
        endCommandBuffer(commandBuffer);
    }

    void Renderer::recordPostprocess() {
        VkCommandBuffer commandBuffer = mPostprocessCommandBuffers[mCurrentFrame];
        vkResetCommandBuffer(commandBuffer, 0);
        beginCommandBuffer(commandBuffer);
        ImageTransitInfo2 transitInfo {
            .commandBuffer = commandBuffer,
            .image = mRasterInstanceImage.get(),
            .srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
            .srcStage = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            .dstStage = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .newLayout = VK_IMAGE_LAYOUT_GENERAL,
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
        };
        Image::transit(transitInfo);
        const PostprocessPassRecordInfo recordInfo {
            .commandBuffer = commandBuffer,
            .extent = mSwapchain.extent(),
            .currentFrame = mCurrentFrame,
            .constants = {
                .exposure = mModel->settings().exposure,
                .tonemapMode = static_cast<uint32_t>(mModel->settings().tonemapMode),
                .displayMode = static_cast<uint32_t>(mModel->settings().displayMode),
                .renderScale = mEffectiveScale,
                .autoExposure = mModel->settings().autoExposure ? 1u : 0u,
                .deltaTime = mDeltaTime,
                .renderWidth = (mSwapchain.extent().width + mEffectiveScale - 1) / mEffectiveScale,
                .renderHeight = (mSwapchain.extent().height + mEffectiveScale - 1) / mEffectiveScale
            }
        };
        mPostprocessPass.record(recordInfo);
        Image::inverseTransit(transitInfo);
    }

    void Renderer::recordPresent(uint32_t imageIndex) {
        VkCommandBuffer commandBuffer = mPostprocessCommandBuffers[mCurrentFrame];
        const ImageTransitInfo2 presentTransitInfo {
            .commandBuffer = commandBuffer,
            .image = mFinalImage.get(),
            .srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT,
            .srcStage = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            .dstStage = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
            .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
        };

        ImageTransitInfo2 swapchainTransitInfo {
            .commandBuffer = commandBuffer,
            .image = mSwapchainImages[imageIndex],
            .srcAccessMask = 0,
            .dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
            .srcStage = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            .dstStage = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
        };
        Image::transit({presentTransitInfo, swapchainTransitInfo});

        auto [width, height]  = mSwapchain.extent();
        VkImageBlit blit{
            .srcSubresource = {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .layerCount = 1
            },
            .srcOffsets = {
                {0, 0, 0},
                {static_cast<int32_t>(width), static_cast<int32_t>(height), 1}
            },
            .dstSubresource = {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .layerCount = 1
            },
            .dstOffsets = {
                {0, 0, 0},
                {static_cast<int32_t>(width), static_cast<int32_t>(height), 1}
            }
        };

        vkCmdBlitImage(
            commandBuffer,
            mFinalImage.get(),
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            mSwapchainImages[imageIndex],
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1,
            &blit,
            VK_FILTER_NEAREST
        );

        swapchainTransitInfo.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
        swapchainTransitInfo.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        swapchainTransitInfo.srcStage = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        Image::inverseTransit({presentTransitInfo, swapchainTransitInfo});

        const AppUIRecordInfo recordInfo {
            .commandBuffer = commandBuffer,
            .imageView = &mSwapchainImageViews[imageIndex]
        };
        mUI.record(recordInfo);
        swapchainTransitInfo.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
        swapchainTransitInfo.dstAccessMask = 0;
        swapchainTransitInfo.srcStage = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        swapchainTransitInfo.dstStage = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        swapchainTransitInfo.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        swapchainTransitInfo.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        Image::transit(swapchainTransitInfo);
        endCommandBuffer(commandBuffer);
    }

    void Renderer::recordPixelRead(VkCommandBuffer commandBuffer) {
        if (mClickedPixel.x == UINT32_MAX) return;
        const ImageTransitInfo2 transitInfo {
            .commandBuffer = commandBuffer,
            .image = mTracerInstanceImage.get(),
            .srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT,
            .srcStage = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
            .dstStage = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
            .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
        };
        Image::transit(transitInfo);

        const uint32_t scale = mEffectiveScale;
        const VkBufferImageCopy region {
            .bufferOffset = 0,
            .bufferRowLength = 0,
            .bufferImageHeight = 0,
            .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
            .imageOffset = {static_cast<int32_t>(mClickedPixel.x / scale), static_cast<int32_t>(mClickedPixel.y / scale), 0},
            .imageExtent = {1, 1, 1}
        };
        vkCmdCopyImageToBuffer(commandBuffer, mTracerInstanceImage.get(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            mReadbackBuffer.get(),
            1, &region);

        Image::inverseTransit(transitInfo);
        mClickedPixel = {UINT32_MAX, UINT32_MAX};
        mPickPending = true;
    }

    void Renderer::updateSelectedInstance() {
        if (mPickPending) {
            mPickPending = false;
            uint32_t* data = nullptr;
            vmaMapMemory(mContext.allocator(), mReadbackBuffer.allocation(), (void**)&data);
            const uint32_t id = *data;
            vmaUnmapMemory(mContext.allocator(), mReadbackBuffer.allocation());
            mModel->select(id, mAdditiveSelect);
        }
    }

    void Renderer::update() {
        const RayTracerPassUpdateInfo tracerUpdateInfo {
            .camera = mModel->camera(),
            .directLight = mModel->scene().directLight(),
            .currentFrame = mCurrentFrame
        };
        mRayTracerPass.update(tracerUpdateInfo);

        const RasterizerPassUpdateInfo rasterUpdateInfo {
            .camera = mModel->camera(),
            .currentFrame = mCurrentFrame
        };
        mRasterizerPass.update(rasterUpdateInfo);
    }

    void Renderer::drawControlPanel() {
        const ExposureReadback readback = mPostprocessPass.exposure();
        const AppUIDrawInfo drawInfo {
            .drawUI = mRenderImGui,
            .camera = mModel->camera(),
            .selectedInstances = &mModel->selection().selectedInstances,
            .activeInstance = mModel->selection().activeInstance,
            .frameCount = mFrameCount,
            .renderScale = mEffectiveScale,
            .autoExposureValue = readback.exposure,
            .avgLuminance = readback.avgLuminance
        };
        mUI.draw(drawInfo);
    }

    void Renderer::flushUpdates() {
        UpdateState& state = mModel->updateState();
        if (!state.any()) return;
        if (state.heavy()) vkDeviceWaitIdle(mContext.device());

        if (state.addModel) {
            mResourceManager.addModel();
            mRayTracerPass.bindTextures();
            mRayTracerPass.rebindScene();
            mFrameCount = 0;
            state.clear();
            return;
        }

        for (const uint32_t sourceIndex : state.dirtyTextures) {
            const uint32_t index = mResourceManager.uploadTexture(sourceIndex);
            mRayTracerPass.bindTexture(index);
        }
        if (state.updateSkybox) {
            const uint32_t skyboxIndex = mModel->scene().skyboxIndex();
            if (skyboxIndex == UINT32_MAX) {
                mResourceManager.disableSkybox();
            } else {
                mResourceManager.uploadSkybox(skyboxIndex);
                mRayTracerPass.bindTexture(skyboxIndex);
            }
        }

        if (state.updateInstances) {
            mResourceManager.rebuildInstances();
            mRayTracerPass.bindInstances();
        } else if (!state.dirtyInstances.empty()) {
            bool tlasDirty = false;
            for (const auto& [index, update] : state.dirtyInstances) {
                if (update == InstanceUpdate::Model) {
                    mResourceManager.updateInstance(index);
                    tlasDirty = true;
                } else {
                    mResourceManager.updateInstanceData(index);
                }
            }
            if (tlasDirty) mResourceManager.refreshTLAS();
            mResourceManager.updateEmissiveIndices();
        }

        if (state.updateMaterials) {
            mResourceManager.rebuildMaterials();
            mRayTracerPass.bindMaterials();
        } else {
            for (const uint32_t index : state.dirtyMaterials) mResourceManager.updateMaterial(index);
        }

        mFrameCount = 0;
        state.clear();
    }

    void Renderer::record(const uint32_t imageIndex) {
        recordTracer();
        recordRaster();
        recordPostprocess();
        recordPresent(imageIndex);
        vkResetFences(mContext.device(), 1, &mFences[mCurrentFrame].get());
    }

    void Renderer::submit(const uint32_t imageIndex) {
        VkPipelineStageFlags tracerWaitStages[] = {VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR};
        const VkSubmitInfo tracerSubmitInfo{
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &mImageAvailableSemaphores[mCurrentFrame].get(),
            .pWaitDstStageMask = tracerWaitStages,
            .commandBufferCount = 1,
            .pCommandBuffers = &mTracerCommandBuffers[mCurrentFrame],
            .signalSemaphoreCount = 1,
            .pSignalSemaphores = &mTracerFinishedSemaphores[mCurrentFrame].get()
        };
        VkQueue queue = mContext.queue(QueueFamilyType::GRAPHICS);
        if (vkQueueSubmit(queue, 1, &tracerSubmitInfo, nullptr) != VK_SUCCESS) {
            throw std::runtime_error("Failed to submit draw command buffer!");
        }

        VkPipelineStageFlags rasterWaitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
        const VkSubmitInfo rasterSubmitInfo{
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &mTracerFinishedSemaphores[mCurrentFrame].get(),
            .pWaitDstStageMask = rasterWaitStages,
            .commandBufferCount = 1,
            .pCommandBuffers = &mRasterCommandBuffers[mCurrentFrame],
            .signalSemaphoreCount = 1,
            .pSignalSemaphores = &mRasterFinishedSemaphores[mCurrentFrame].get()
        };
        if (vkQueueSubmit(queue, 1, &rasterSubmitInfo, nullptr) != VK_SUCCESS) {
            throw std::runtime_error("Failed to submit draw command buffer!");
        }

        VkPipelineStageFlags postprocessWaitStages[] = {VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT};
        const VkSubmitInfo postprocessSubmitInfo{
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &mRasterFinishedSemaphores[mCurrentFrame].get(),
            .pWaitDstStageMask = postprocessWaitStages,
            .commandBufferCount = 1,
            .pCommandBuffers = &mPostprocessCommandBuffers[mCurrentFrame],
            .signalSemaphoreCount = 1,
            .pSignalSemaphores = &mPostprocessFinishedSemaphores[imageIndex].get()
        };
        if (vkQueueSubmit(queue, 1, &postprocessSubmitInfo, mFences[mCurrentFrame].get()) != VK_SUCCESS) {
            throw std::runtime_error("Failed to submit draw command buffer!");
        }

        const VkPresentInfoKHR presentInfo {
            .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &mPostprocessFinishedSemaphores[imageIndex].get(),
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

    void Renderer::acquireNextImage(uint32_t& imageIndex) {
        SwapchainAcquireInfo swapchainAcquireInfo {
            .imageAvailableSemaphore = mImageAvailableSemaphores[mCurrentFrame].get(),
            .fence = VK_NULL_HANDLE,
            .imageIndex = &imageIndex
        };
        const VkResult result = mSwapchain.acquireNextImage(swapchainAcquireInfo);
        if (result != VK_SUCCESS) {
            throw std::runtime_error("Failed to acquire image!");
        }
    }

    void Renderer::beginFrame() {
        vkWaitForFences(mContext.device(), 1, &mFences[mCurrentFrame].get(), VK_TRUE, UINT64_MAX);
        updateSelectedInstance();
        drawControlPanel();
    }

    void Renderer::endFrame() {
        const auto now = std::chrono::steady_clock::now();
        mDeltaTime = std::chrono::duration<float>(now - mLastFrameTime).count();
        mLastFrameTime = now;
        const bool cameraMoved = mModel->updateState().cameraMoved;
        flushUpdates();
        uint32_t imageIndex;
        acquireNextImage(imageIndex);

        const uint32_t scale = cameraMoved ? mModel->settings().effectiveMotionScale() : mModel->settings().effectiveRenderScale();
        if (scale != mEffectiveScale) {
            mEffectiveScale = scale;
            mFrameCount = 0;
        }

        update();
        record(imageIndex);
        submit(imageIndex);
        updateCurrentFrame();
        ++mFrameCount;
    }

    void Renderer::pick(const glm::dvec2 cursor, const bool additive) {
        GLFWwindow* window = mContext.window().glfwWindow();
        int winWidth, winHeight, fbWidth, fbHeight;
        glfwGetWindowSize(window, &winWidth, &winHeight);
        glfwGetFramebufferSize(window, &fbWidth, &fbHeight);
        if (winWidth <= 0 or winHeight <= 0) return;
        const double x = cursor.x * static_cast<double>(fbWidth)  / winWidth;
        const double y = cursor.y * static_cast<double>(fbHeight) / winHeight;
        auto [width, height] = mSwapchain.extent();
        if (x < 0.0 or y < 0.0 or x >= width or y >= height) return;
        mClickedPixel = {static_cast<uint32_t>(x), static_cast<uint32_t>(y)};
        mAdditiveSelect = additive;
    }

    void Renderer::saveImage() {
        vkDeviceWaitIdle(mContext.device());
        auto [width, height] = mSwapchain.extent();

        ImageCreateInfo saveImageInfo {
            .device = mContext.device(),
            .allocator = mContext.allocator(),
            .flags = 0,
            .format = VK_FORMAT_R8G8B8A8_SRGB,
            .extent = {width, height, 1},
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .tiling = VK_IMAGE_TILING_OPTIMAL,
            .imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
            .memoryUsage = VMA_MEMORY_USAGE_AUTO
        };
        Image saveImage(saveImageInfo);

        const VkDeviceSize bufferSize = static_cast<VkDeviceSize>(width) * height * 4;
        const BufferCreateInfo bufferInfo {
            .allocator = mContext.allocator(),
            .size = bufferSize,
            .bufferUsage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .memoryUsage = VMA_MEMORY_USAGE_CPU_ONLY
        };
        Buffer buffer(bufferInfo);

        auto [commandBuffer, cmdData] = beginCommandBuffer(mContext.device(),
                                            mContext.familyIndex(QueueFamilyType::GRAPHICS).value());

        const ImageTransitInfo2 finalTransitInfo {
            .commandBuffer = commandBuffer,
            .image = mFinalImage.get(),
            .srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT,
            .srcStage = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            .dstStage = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
            .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
        };
        ImageTransitInfo2 saveTransitInfo {
            .commandBuffer = commandBuffer,
            .image = saveImage.get(),
            .srcAccessMask = VK_ACCESS_2_NONE,
            .dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
            .srcStage = VK_PIPELINE_STAGE_2_NONE,
            .dstStage = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
        };
        Image::transit({finalTransitInfo, saveTransitInfo});

        const VkImageBlit blit {
            .srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
            .srcOffsets = {{0, 0, 0}, {static_cast<int32_t>(width), static_cast<int32_t>(height), 1}},
            .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
            .dstOffsets = {{0, 0, 0}, {static_cast<int32_t>(width), static_cast<int32_t>(height), 1}}
        };
        vkCmdBlitImage(commandBuffer,
            mFinalImage.get(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            saveImage.get(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1, &blit, VK_FILTER_NEAREST);

        saveTransitInfo.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        saveTransitInfo.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
        saveTransitInfo.srcStage = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        saveTransitInfo.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        saveTransitInfo.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        Image::transit(saveTransitInfo);

        const VkBufferImageCopy region {
            .bufferOffset = 0,
            .bufferRowLength = 0,
            .bufferImageHeight = 0,
            .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
            .imageOffset = {0, 0, 0},
            .imageExtent = {width, height, 1}
        };
        vkCmdCopyImageToBuffer(commandBuffer, saveImage.get(),
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buffer.get(), 1, &region);

        Image::inverseTransit(finalTransitInfo);
        endCommandBuffer(cmdData, mContext.queue(QueueFamilyType::GRAPHICS));

        const fs::path outputDir = fs::path(PROJECT_PATH) / "screenshots";
        fs::create_directories(outputDir);
        const auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        char stamp[32];
        std::strftime(stamp, sizeof(stamp), "%Y%m%d_%H%M%S", std::localtime(&now));
        const std::string path = (outputDir / ("render_" + std::string(stamp) + ".png")).string();

        uint8_t* data = nullptr;
        vmaMapMemory(mContext.allocator(), buffer.allocation(), reinterpret_cast<void**>(&data));
        const int ok = stbi_write_png(path.c_str(), static_cast<int>(width), static_cast<int>(height),
                                      4, data, static_cast<int>(width) * 4);
        vmaUnmapMemory(mContext.allocator(), buffer.allocation());

        if (ok) INFO << "Saved image: " << path;
        else    ERROR << "Failed to save image: " << path;
    }
}
