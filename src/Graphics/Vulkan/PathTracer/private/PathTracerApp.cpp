//
// Created by igor on 4/22/26.
//

#include "PathTracerApp.hpp"
#include "CoreUtils.hpp"
#include "Timer.hpp"
#include "Loader.hpp"
#include "BinnedSAHBuilder.hpp"
#include "CallBacks.hpp"

#define COLOR_FORMAT VK_FORMAT_R8G8B8A8_UNORM
#define NORMAL_FORMAT VK_FORMAT_R8G8B8A8_SNORM

namespace crv::graphics::vulkan {
    PathTracerApp::PathTracerApp(const PathTracerAppCreateInfo& info) {
        createCamera(info.cameraCreateInfo);
        mDirectLight = info.directLight;
        createContext(info.windowCreateInfo);
        setCallBacks(this);
        createCommandPool();
        createCommandBuffers();
        createSwapChain();
        createSwapChainImages();
        createSyncObjects();
        createPresentImage();

        loadModel(info);
        createTextures();

        createGBuffers();
        createRasterizer();
        createPathTracer();
        createImGui();
    }

    void PathTracerApp::run() {
        utils::FpsCounter fpsCounter;
        double deltaTime = 0;
        const Window& window = mContext.window();
        while (!window.shouldClose()) {
            glfwPollEvents();
            window.keyboardCallBack(deltaTime);
            fpsCounter.update();
            deltaTime = 1e3 / fpsCounter.fps();
            window.setTitle(std::to_string(fpsCounter.fps()).c_str());
            drawFrame();
            updateCurrentFrame();
            ++mFrameCount;
        }
        vkDeviceWaitIdle(mContext.device());
    }

    void PathTracerApp::createCamera(const scene::CameraCreateInfo& info) {
        mFlyCamera = scene::FlyCamera(info);
        mOrbitalCamera = scene::OrbitalCamera(info);
        if (info.type == scene::CameraType::FLY) {
            mCamera = &mFlyCamera;
        } else {
            mCamera = &mOrbitalCamera;
        }
    }

    void PathTracerApp::createContext(const WindowCreateInfo& windowCreateInfo) {
        const ContextCreateInfo createInfo {
            .windowCreateInfo = windowCreateInfo,
            .validationLayers = { "VK_LAYER_KHRONOS_validation" },
            .deviceExtensions = { VK_KHR_SWAPCHAIN_EXTENSION_NAME,
                                     VK_KHR_MAINTENANCE_1_EXTENSION_NAME },
            .enableValidationLayers = mDebug
        };
        mContext = Context(createInfo);
    }

    void PathTracerApp::createCommandPool() {
        CommandPoolCreateInfo createInfo {
            .device = mContext.device(),
            .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            .queueFamilyIndex = mContext.familyIndex(QueueFamilyType::COMPUTE).value()
        };
        mComputeCommandPool = CommandPool(createInfo);
        createInfo.queueFamilyIndex = mContext.familyIndex(QueueFamilyType::GRAPHICS).value();
        mGraphicsCommandPool = CommandPool(createInfo);
    }

    void PathTracerApp::createCommandBuffers() {
        CommandBuffersCreateInfo createInfo {
            .device = mContext.device(),
            .commandPool = mComputeCommandPool.get(),
            .bufferCount = mFramesInFlight
        };
        mComputeCommandBuffers = CommandBuffers(createInfo);
        createInfo.commandPool = mGraphicsCommandPool.get();
        mGraphicsCommandBuffers = CommandBuffers(createInfo);
    }

    void PathTracerApp::createSwapChain() {
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

    void PathTracerApp::createSwapChainImages() {
        auto [capabilities, formats, presentModes] = Swapchain::getSupport(mContext.physicalDevice(), mContext.surface());
        uint32_t imageCount = Swapchain::getImageCount(capabilities);
        vkGetSwapchainImagesKHR(mContext.device(), mSwapchain.get(), &imageCount, nullptr);
        mSwapchainImages.resize(imageCount);
        vkGetSwapchainImagesKHR(mContext.device(), mSwapchain.get(), &imageCount, mSwapchainImages.data());

        auto [commandPool, commandBuffers] = beginCommandBuffer(mContext.device(), mContext.familyIndex(QueueFamilyType::COMPUTE).value());
        VkCommandBuffer commandBuffer = (*commandBuffers)[0];
        for (uint32_t i = 0; i < imageCount; ++i) {
            const ImageViewCreateInfo imageViewCreateInfo{
                .device = mContext.device(),
                .image = mSwapchainImages[i],
                .viewType = VK_IMAGE_VIEW_TYPE_2D,
                .format = mSwapchain.format(),
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .mipLevels = 1,
                .baseMipLevel = 0,
                .baseArrayLayer = 0,
                .layerCount = 1
            };
            mSwapchainImageViews.emplace_back(imageViewCreateInfo);
            const ImageTransitInfo imageTransitInfo {
                .commandBuffer = commandBuffer,
                .image = mSwapchainImages[i],
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

    void PathTracerApp::createSyncObjects() {
        const SemaphoreCreateInfo semaphoreCreateInfo {
            .device = mContext.device()
        };
        const FenceCreateInfo fenceCreateInfo {
            .device = mContext.device()
        };
        mImageAvailableSemaphores.resize(mFramesInFlight);
        mRasterFinishedSemaphores.resize(mFramesInFlight);
        mTracerFinishedSemaphores.resize(mFramesInFlight);
        mFences.resize(mFramesInFlight);
        for (uint32_t i = 0; i < mFramesInFlight; ++i) {
            mImageAvailableSemaphores[i] = Semaphore(semaphoreCreateInfo);
            mRasterFinishedSemaphores[i] = Semaphore(semaphoreCreateInfo);
            mTracerFinishedSemaphores[i] = Semaphore(semaphoreCreateInfo);
            mFences[i] = Fence(fenceCreateInfo);
        }
    }

    void PathTracerApp::createPresentImage() {
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
        mPresentImage = Image(imageCreateInfo);

        const ImageViewCreateInfo imageViewCreateInfo {
            .device = mContext.device(),
            .image = mPresentImage.get(),
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format = VK_FORMAT_R8G8B8A8_UNORM,
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .mipLevels = 1,
            .baseMipLevel = 0,
            .baseArrayLayer = 0,
            .layerCount = 1
        };
        mPresentImageView = ImageView(imageViewCreateInfo);
    }

    void PathTracerApp::createTextures() {
        mTextures.resize(mMaterials.size());
        for (size_t i = 0; i < mMaterials.size(); ++i) {
            const cm::Material& material = mMaterials[i];
            TexturesByType& texturesByType = mTextures[i];
            for (int texType = 0; texType < static_cast<int>(cm::Texture::UNKNOWN); ++texType) {
                const cm::Texture& texture = material.mTextures[texType];
                TextureCreateInfo textureCreateInfo {
                    .device = mContext.device(),
                    .physicalDevice = mContext.physicalDevice(),
                    .allocator = mContext.allocator(),
                    .queue = mContext.queue(QueueFamilyType::COMPUTE),
                    .queueFamilyIndex = mContext.familyIndex(QueueFamilyType::COMPUTE).value(),
                    .dataByLevel = texture.mDataByLevel,
                    .texFormat = texture.mFormat,
                    .mipLevels = 1,
                    .arrayLayers = 1,
                    .samples = VK_SAMPLE_COUNT_1_BIT,
                    .tiling = VK_IMAGE_TILING_OPTIMAL,
                    .memoryUsage = VMA_MEMORY_USAGE_AUTO
                };
                texturesByType[texType] = Texture(textureCreateInfo);
            }
        }
    }

    static BVH buildBVH(std::span<Tri> tris) {
        utils::Timer timer;
        timer.start();
        BinnedSAHBuilder<BVH::Node, Tri> builder{ tris };
        BVH bvh = builder.build();
        INFO << "BVH build time: " << timer.duration() / 1000 << " sec";
        return bvh;
    }

    void PathTracerApp::loadModel(const PathTracerAppCreateInfo& info) {
        auto loader = new cm::Loader;
        utils::Timer timer;
        timer.start();
        loader->setModel(info.modelPath);
        loader->load(info.modelMatrix);
        INFO << "Model load time: " << timer.duration() / 1000 << " sec";

        timer.start();
        const auto& lIndices = loader->indices();
        const auto& lVertices = loader->vertices();
        const auto& lMeshes = loader->meshes();
        std::vector<Tri> triangles;
        std::vector<uint32_t> indices;
        std::vector<uint32_t> materialIndices;
        for (size_t i = 0; i < lMeshes.size(); ++i) {
            const auto& mesh = lMeshes[i];
            for (size_t j = 0; j < mesh.numIndices; j += 3) {
                const size_t idx = mesh.baseIndex + j;
                triangles.emplace_back(lVertices[lIndices[idx + 0]].pos,
                                       lVertices[lIndices[idx + 1]].pos,
                                       lVertices[lIndices[idx + 2]].pos);
                indices.emplace_back(lIndices[idx + 0]);
                indices.emplace_back(lIndices[idx + 1]);
                indices.emplace_back(lIndices[idx + 2]);
                materialIndices.emplace_back(mesh.materialIndex);
            }

            for (size_t j = 0; j < mesh.numVertices; ++j) {
                const cm::Vertex& modelVertex = lVertices[mesh.baseVertex + j];
                Vertex vertex{
                    .pos = modelVertex.pos,
                    .texCoord = modelVertex.texCoord0,
                    .normal = modelVertex.normal,
                    .tangent = modelVertex.tangent,
                    .texIndex = static_cast<uint32_t>(mesh.materialIndex * cm::Texture::UNKNOWN)
                };
                mVertices.push_back(vertex);
            }
        }
        INFO << "Primitive creation time: " << timer.duration() / 1000 << " sec";
        INFO << "Total number of primitives: " << triangles.size();
        auto bvh = buildBVH(std::span(triangles));
        //reorder after bvh
        for (int i = 0; i < triangles.size(); ++i) {
            const uint32_t idx = bvh.primIds()[i];
            Tri& tri = triangles[idx];
            mTriangles.emplace_back(Vec4(tri.p0, 1), Vec4(tri.e1, 1),
                                          Vec4(tri.e2, 1), Vec4(tri.N, 1));
            cm::Vertex v0 = lVertices[indices[idx * 3 + 0]];
            cm::Vertex v1 = lVertices[indices[idx * 3 + 1]];
            cm::Vertex v2 = lVertices[indices[idx * 3 + 2]];
            mTriangleExtras.emplace_back(v0.texCoord0, v1.texCoord0, v2.texCoord0);
            mMaterialIndices.push_back(materialIndices[idx]);
        }
        mIndices = loader->indices();

        for (const auto& node: bvh.nodes()) {
            AlignedNode alignedNode{};
            alignedNode.bbox = AlignedBBox(Vec4(node.bbox().min, 1), Vec4(node.bbox().max, 1));
            alignedNode.index = node.index().value();
            mNodes.push_back(alignedNode);
        }
        mMaterials = loader->materials();
    }

    void PathTracerApp::createPathTracer() {
        const PathTracerCreateInfo pathTracerCreateInfo {
            .context = &mContext,
            .triangles = mTriangles,
            .triangleExtras = mTriangleExtras,
            .nodes = mNodes,
            .textures = &mTextures,
            .materialIndices = mMaterialIndices,
            .framesInFlight = mFramesInFlight
        };
        mPathTracer = PathTracer(pathTracerCreateInfo);
    }

    void PathTracerApp::createImGui() {
        const ImGuiCreateInfo createInfo {
            .context = &mContext,
            .imageCount = static_cast<uint32_t>(mSwapchainImages.size()),
            .format = mSwapchain.format()
        };
        mImGui = VkImGui(createInfo);
    }

    void PathTracerApp::update() {
        mDirectLight.dir = glm::normalize(mDirectLight.dir);
        const RasterizerUpdateInfo rasterizerUpdateInfo {
            .camera = mCamera,
            .currentFrame = mCurrentFrame
        };
        mRasterizer.update(rasterizerUpdateInfo);
        const PathTracerUpdateInfo pathTracerUpdateInfo {
            .camera = mCamera,
            .directLight = mDirectLight,
            .gBuffer = &mGBuffers[mCurrentFrame],
            .presentImage = mPresentImage.get(),
            .presentImageView = mPresentImageView.get(),
            .currentFrame = mCurrentFrame
        };
        mPathTracer.update(pathTracerUpdateInfo);
    }

    void PathTracerApp::recordRaster() {
        VkCommandBuffer commandBuffer = mGraphicsCommandBuffers[mCurrentFrame];
        vkResetCommandBuffer(commandBuffer, 0);
        beginCommandBuffer(commandBuffer);

        const RasterizerRecordInfo rasterizerRecordInfo {
            .commandBuffer = commandBuffer,
            .gBuffer = &mGBuffers[mCurrentFrame],
            .extent = mSwapchain.extent(),
            .currentFrame = mCurrentFrame
        };
        mRasterizer.record(rasterizerRecordInfo);

        GBuffer& gBuffer = mGBuffers[mCurrentFrame];
        const ImageBarrierInfo colorBarrierInfo {
            .image = gBuffer.colorImage.get(),
            .srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
        };
        const ImageBarrierInfo depthBarrierInfo {
            .image = gBuffer.depthImage.get(),
            .srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            .aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
        };
        ImageBarrierInfo normalBarrierInfo = colorBarrierInfo;
        normalBarrierInfo.image = gBuffer.normalImage.get();
        const std::vector barriers = {Image::barrier(colorBarrierInfo), Image::barrier(depthBarrierInfo),
                                      Image::barrier(normalBarrierInfo)};
        const ImagePipelineBarrierInfo pipelineBarrierInfo {
            .commandBuffer = commandBuffer,
            .srcStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
            .dstStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            .barriers = barriers
        };
        Image::pipelineBarrier(pipelineBarrierInfo);
        endCommandBuffer(commandBuffer);
    }

     void PathTracerApp::recordPresent(uint32_t imageIndex, VkCommandBuffer commandBuffer) {
        const ImageBarrierInfo dataBarrierInfo {
            .image = mPresentImage.get(),
            .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
            .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
        };
        const ImageBarrierInfo presentBarrierInfo {
            .image = mSwapchainImages[imageIndex],
            .srcAccessMask = 0,
            .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
            .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
        };
        const VkImageMemoryBarrier dataBarrier    = Image::barrier(dataBarrierInfo);
        const VkImageMemoryBarrier presentBarrier = Image::barrier(presentBarrierInfo);

        ImagePipelineBarrierInfo pipelineBarrierInfo1 {
            .commandBuffer = commandBuffer,
            .srcStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            .dstStage = VK_PIPELINE_STAGE_TRANSFER_BIT,
            .barriers = {dataBarrier, presentBarrier}
        };
        Image::pipelineBarrier(pipelineBarrierInfo1);

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
            mPresentImage.get(),
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            mSwapchainImages[imageIndex],
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1,
            &blit,
            VK_FILTER_NEAREST
        );

        VkImageMemoryBarrier inversePresentBarrier = Image::inverseBarrier(presentBarrier);

        VkPipelineStageFlags dstStage = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
        if (mRenderImGui) {
            inversePresentBarrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            inversePresentBarrier.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            dstStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        }
        ImagePipelineBarrierInfo pipelineBarrierInfo2 {
            .commandBuffer = commandBuffer,
            .srcStage = VK_PIPELINE_STAGE_TRANSFER_BIT,
            .dstStage = dstStage,
            .barriers = {inversePresentBarrier}
        };
        Image::pipelineBarrier(pipelineBarrierInfo2);

        if (mRenderImGui) {
            ImGuiRenderInfo renderInfo {
                .commandBuffer = commandBuffer,
                .imageView = mSwapchainImageViews[imageIndex].get(),
                .extent = mSwapchain.extent()
            };
            mImGui.render(renderInfo);
            inversePresentBarrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            inversePresentBarrier.dstAccessMask = 0;
            inversePresentBarrier.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            inversePresentBarrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

            ImagePipelineBarrierInfo guiBarrierInfo {
                .commandBuffer = commandBuffer,
                .srcStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                .dstStage = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                .barriers = {inversePresentBarrier}
            };
            Image::pipelineBarrier(guiBarrierInfo);
        }
    }

    void PathTracerApp::recordTracer(uint32_t imageIndex) {
        VkCommandBuffer commandBuffer = mComputeCommandBuffers[mCurrentFrame];
        vkResetCommandBuffer(commandBuffer, 0);
        beginCommandBuffer(commandBuffer);

        const PathTracerRecordInfo pathTracerRecordInfo {
            .commandBuffer = commandBuffer,
            .extent = mSwapchain.extent(),
            .currentFrame = mCurrentFrame,
            .frameCount = mFrameCount,
            .maxDepth = static_cast<uint32_t>(mMaxDepth)
        };
        mPathTracer.record(pathTracerRecordInfo);

        GBuffer& gBuffer = mGBuffers[mCurrentFrame];
        const ImageBarrierInfo colorBarrierInfo {
            .image = gBuffer.colorImage.get(),
            .srcAccessMask = VK_ACCESS_SHADER_READ_BIT,
            .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            .newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
        };
        const ImageBarrierInfo depthBarrierInfo {
            .image = gBuffer.depthImage.get(),
            .srcAccessMask = VK_ACCESS_SHADER_READ_BIT,
            .dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            .newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            .aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
        };
        ImageBarrierInfo normalBarrierInfo = colorBarrierInfo;
        normalBarrierInfo.image = gBuffer.normalImage.get();
        const std::vector barriers = {Image::barrier(colorBarrierInfo), Image::barrier(depthBarrierInfo),
                                      Image::barrier(normalBarrierInfo)};
        const ImagePipelineBarrierInfo pipelineBarrierInfo {
            .commandBuffer = commandBuffer,
            .srcStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            .dstStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
            .barriers = barriers
        };
        Image::pipelineBarrier(pipelineBarrierInfo);

        recordPresent(imageIndex, commandBuffer);
        endCommandBuffer(commandBuffer);
    }

    void PathTracerApp::record(uint32_t imageIndex) {
        recordRaster();
        recordTracer(imageIndex);
        vkResetFences(mContext.device(), 1, &mFences[mCurrentFrame].get());
    }

    void PathTracerApp::submit(const uint32_t imageIndex) {
        auto& graphicsCommandBuffer = mGraphicsCommandBuffers[mCurrentFrame];
        VkPipelineStageFlags rasterWaitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
        const VkSubmitInfo rasterSubmitInfo{
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &mImageAvailableSemaphores[mCurrentFrame].get(),
            .pWaitDstStageMask = rasterWaitStages,
            .commandBufferCount = 1,
            .pCommandBuffers = &graphicsCommandBuffer,
            .signalSemaphoreCount = 1,
            .pSignalSemaphores = &mRasterFinishedSemaphores[mCurrentFrame].get()
        };

        VkQueue graphicsQueue = mContext.queue(QueueFamilyType::GRAPHICS);
        if (vkQueueSubmit(graphicsQueue, 1, &rasterSubmitInfo, VK_NULL_HANDLE) != VK_SUCCESS) {
            throw std::runtime_error("Failed to submit draw command buffer!");
        }

        auto& computeCommandBuffer = mComputeCommandBuffers[mCurrentFrame];
        VkPipelineStageFlags tracerWaitStages[] = {VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT};
        const VkSubmitInfo tracerSubmitInfo{
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &mRasterFinishedSemaphores[mCurrentFrame].get(),
            .pWaitDstStageMask = tracerWaitStages,
            .commandBufferCount = 1,
            .pCommandBuffers = &computeCommandBuffer,
            .signalSemaphoreCount = 1,
            .pSignalSemaphores = &mTracerFinishedSemaphores[mCurrentFrame].get()
        };
        VkQueue computeQueue = mContext.queue(QueueFamilyType::COMPUTE);
        if (vkQueueSubmit(computeQueue, 1, &tracerSubmitInfo, mFences[mCurrentFrame].get()) != VK_SUCCESS) {
            throw std::runtime_error("Failed to submit draw command buffer!");
        }

        const VkPresentInfoKHR presentInfo {
            .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &mTracerFinishedSemaphores[mCurrentFrame].get(),
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

    void PathTracerApp::acquireNextImage(uint32_t& imageIndex) {
        SwapchainAcquireInfo swapchainAcquireInfo {
            .imageAvailableSemaphore = mImageAvailableSemaphores[mCurrentFrame].get(),
            .fence = mFences[mCurrentFrame].get(),
            .imageIndex = &imageIndex
        };
        const VkResult result = mSwapchain.acquireNextImage(swapchainAcquireInfo);
        if (result != VK_SUCCESS) {
            throw std::runtime_error("Failed to acquire image!");
        }
    }

    void PathTracerApp::drawFrame() {
        uint32_t imageIndex;
        acquireNextImage(imageIndex);
        drawControlPanel();
        update();
        record(imageIndex);
        submit(imageIndex);
    }

    void PathTracerApp::drawControlPanel() {
        ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(500, 200), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowBgAlpha(0.4);
        mImGui.beginFrame();
        if (!mRenderImGui) {
            mImGui.endFrame();
            return;
        }
        const bool isFlyCamera = mCamera->type() == cs::CameraType::FLY;
        ImGui::Begin("Settings");
        const ImVec2 mousePos = ImGui::GetMousePos();
        ImGui::Text("Mouse pos: %.1f x %.1f", mousePos.x, mousePos.y);
        ImGui::Text("FPS: %.1f", ImGui::GetIO().Framerate);

        VkImGui::beginGroup("PathTracer Settings");
        if (ImGui::DragInt("Max Depth", &mMaxDepth, 0.05f, 1, 10)) {
            updateImage();
        }
        VkImGui::endGroup();

        VkImGui::beginGroup("Direct Light");
        if (ImGui::DragFloat3("Direction", &mDirectLight.dir.x, 0.005f, -1.0f, 1.0f)) {
            updateImage();
        }

        if (ImGui::DragFloat("Intensity", &mDirectLight.intensity, 0.05f, 0.0f, 10.0f)) {
            updateImage();
        }
        VkImGui::endGroup();

        ImGui::Separator();
        if (VkImGui::selectableButton("Fly", isFlyCamera)) {
            setCamera(scene::CameraType::FLY);
        }
        ImGui::SameLine(0.0f, 5.0f);
        if (VkImGui::selectableButton("Orbital", !isFlyCamera)) {
            setCamera(scene::CameraType::ORBITAL);
        }
        ImGui::SameLine();
        ImGui::Text("Camera type");
        ImGui::Separator();
        ImGui::End();
        mImGui.endFrame();
    }

    void PathTracerApp::setCamera(const scene::CameraType type) {
        if (type == scene::CameraType::FLY) {
            mCamera = &mFlyCamera;
            mCamera->setPosition(mOrbitalCamera.position());
            mCamera->setOrientation(mOrbitalCamera.orientation());
        } else {
            mCamera = &mOrbitalCamera;
        }
    }

    void PathTracerApp::createGBuffers() {
        const auto [width, height] = mSwapchain.extent();
        VkExtent3D extent = {width, height, 1};
        const ImageCreateInfo colorImageCreateInfo {
            .device = mContext.device(),
            .allocator = mContext.allocator(),
            .format = COLOR_FORMAT,
            .extent = extent,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .tiling = VK_IMAGE_TILING_OPTIMAL,
            .imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            .memoryUsage = VMA_MEMORY_USAGE_AUTO
        };
        ImageViewCreateInfo colorViewCreateInfo {
            .device = mContext.device(),
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format = COLOR_FORMAT,
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
        };
        const ImageCreateInfo depthImageCreateInfo {
            .device = mContext.device(),
            .allocator = mContext.allocator(),
            .format = VK_FORMAT_D32_SFLOAT,
            .extent = extent,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .tiling = VK_IMAGE_TILING_OPTIMAL,
            .imageUsage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            .memoryUsage = VMA_MEMORY_USAGE_AUTO
        };
        ImageViewCreateInfo depthViewCreateInfo {
            .device = mContext.device(),
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format = VK_FORMAT_D32_SFLOAT,
            .aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
        };
        const ImageCreateInfo normalImageCreateInfo {
            .device = mContext.device(),
            .allocator = mContext.allocator(),
            .format = NORMAL_FORMAT,
            .extent = extent,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .tiling = VK_IMAGE_TILING_OPTIMAL,
            .imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            .memoryUsage = VMA_MEMORY_USAGE_AUTO
        };
        ImageViewCreateInfo normalViewCreateInfo {
            .device = mContext.device(),
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format = NORMAL_FORMAT,
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
        };
        const SamplerCreateInfo samplerCreateInfo {
            .device = mContext.device(),
            .physicalDevice = mContext.physicalDevice(),
            .addressMode = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            .compareEnable = VK_FALSE,
            .compareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
            .mipLevels = 1,
            .borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK
        };
        constexpr ImageBarrierInfo colorBarrierInfo {
            .srcAccessMask = 0,
            .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
        };
        constexpr ImageBarrierInfo depthBarrierInfo {
            .srcAccessMask = 0,
            .dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                             VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            .aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
        };
        VkImageMemoryBarrier colorBarrier = Image::barrier(colorBarrierInfo);
        VkImageMemoryBarrier depthBarrier = Image::barrier(depthBarrierInfo);
        VkImageMemoryBarrier normalBarrier = colorBarrier;
        std::vector<VkImageMemoryBarrier> colorBarriers;
        std::vector<VkImageMemoryBarrier> depthBarriers;
        mGBuffers.resize(mFramesInFlight);
        for (int i = 0; i < mFramesInFlight; ++i) {
            GBuffer& gBuffer = mGBuffers[i];
            gBuffer.colorImage  = Image(colorImageCreateInfo);
            gBuffer.depthImage  = Image(depthImageCreateInfo);
            gBuffer.normalImage = Image(normalImageCreateInfo);
            colorViewCreateInfo.image  = gBuffer.colorImage.get();
            depthViewCreateInfo.image  = gBuffer.depthImage.get();
            normalViewCreateInfo.image = gBuffer.normalImage.get();
            gBuffer.colorView  = ImageView(colorViewCreateInfo);
            gBuffer.depthView  = ImageView(depthViewCreateInfo);
            gBuffer.normalView = ImageView(normalViewCreateInfo);
            colorBarrier.image  = gBuffer.colorImage.get();
            depthBarrier.image  = gBuffer.depthImage.get();
            normalBarrier.image = gBuffer.normalImage.get();
            colorBarriers.push_back(colorBarrier);
            depthBarriers.push_back(depthBarrier);
            colorBarriers.push_back(normalBarrier);
            gBuffer.sampler = Sampler(samplerCreateInfo);
        }
        auto [commandPool, commandBuffers] = beginCommandBuffer(mContext.device(), mContext.familyIndex(QueueFamilyType::GRAPHICS).value());
        VkCommandBuffer commandBuffer = (*commandBuffers)[0];
        const ImagePipelineBarrierInfo colorPipelineBarrierInfo {
            .commandBuffer = commandBuffer,
            .srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            .dstStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            .barriers = colorBarriers
        };
        const ImagePipelineBarrierInfo depthPipelineBarrierInfo {
            .commandBuffer = commandBuffer,
            .srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            .dstStage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
            .barriers = depthBarriers
        };
        Image::pipelineBarrier(colorPipelineBarrierInfo);
        Image::pipelineBarrier(depthPipelineBarrierInfo);
        endCommandBuffer(commandPool, commandBuffers, mContext.queue(QueueFamilyType::GRAPHICS));
    }

    void PathTracerApp::createRasterizer() {
        const auto [width, height] = mSwapchain.extent();
        const RasterizerCreateInfo createInfo {
            .context = &mContext,
            .colorFormat = COLOR_FORMAT,
            .normalFormat = NORMAL_FORMAT,
            .extent = {width, height, 1},
            .vertices = mVertices,
            .indices = mIndices,
            .textures = &mTextures
        };
        mRasterizer = Rasterizer(createInfo);
    }
}