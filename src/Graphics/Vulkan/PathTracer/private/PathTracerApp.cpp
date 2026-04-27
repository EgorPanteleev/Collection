//
// Created by igor on 4/22/26.
//

#include "PathTracerApp.hpp"
#include "CoreUtils.hpp"
#include "Timer.hpp"
#include "Loader.hpp"
#include "BinnedSAHBuilder.hpp"
#include "CallBacks.hpp"

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
        createPathTracer();
        createImGui();
        createRasterizer();
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
        mComputeFinishedSemaphores.resize(mFramesInFlight);
        mFences.resize(mFramesInFlight);
        for (uint32_t i = 0; i < mFramesInFlight; ++i) {
            mImageAvailableSemaphores[i]  = Semaphore(semaphoreCreateInfo);
            mComputeFinishedSemaphores[i] = Semaphore(semaphoreCreateInfo);
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
            .imageUsage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
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

        VkImageMemoryBarrier toColor{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .srcAccessMask = 0,
            .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .image = mPresentImage.get(),
            .subresourceRange = {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1
            }
        };

        auto [commandPool, commandBuffers] = beginCommandBuffer(mContext.device(), mContext.familyIndex(QueueFamilyType::GRAPHICS).value());
        VkCommandBuffer commandBuffer = (*commandBuffers)[0];

        vkCmdPipelineBarrier(
            commandBuffer,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            0,
            0,nullptr,
            0,nullptr,
            1,&toColor
        );

        endCommandBuffer(commandPool, commandBuffers, mContext.queue(QueueFamilyType::GRAPHICS));
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
        // const PathTracerUpdateInfo pathTracerUpdateInfo {
        //     .camera = mCamera,
        //     .directLight = mDirectLight,
        //     .presentImage = mPresentImage.get(),
        //     .presentImageView = mPresentImageView.get(),
        //     .currentFrame = mCurrentFrame
        // };
        // mPathTracer.update(pathTracerUpdateInfo);
    }

    void PathTracerApp::record(uint32_t imageIndex) {
        VkCommandBuffer commandBuffer = mGraphicsCommandBuffers[mCurrentFrame];
        vkResetCommandBuffer(commandBuffer, 0);
        constexpr VkCommandBufferBeginInfo beginInfo {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = 0,
            .pInheritanceInfo = nullptr
        };
        if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
            throw std::runtime_error("Failed to begin recording command buffer!");
        }

        VkImageMemoryBarrier toColor{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .srcAccessMask = 0,
            .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .image = mPresentImage.get(),
            .subresourceRange = {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1
            }
        };

        vkCmdPipelineBarrier(
            commandBuffer,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            0, 0,nullptr, 0,nullptr,
            1, &toColor
        );

        RasterizerRecordInfo rasterizerRecordInfo {
            .commandBuffer = commandBuffer,
            .imageView = mPresentImageView.get(),
            .extent = mSwapchain.extent(),
            .currentFrame = mCurrentFrame
        };
        mRasterizer.record(rasterizerRecordInfo);

        // PathTracerRecordInfo pathTracerRecordInfo {
        //     .commandBuffer = commandBuffer,
        //     .extent = mSwapchain.extent(),
        //     .currentFrame = mCurrentFrame,
        //     .frameCount = mFrameCount,
        //     .maxDepth = static_cast<uint32_t>(mMaxDepth)
        // };
        // mPathTracer.record(pathTracerRecordInfo);
        
        const VkImageMemoryBarrier dataBarrier{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            .image = mPresentImage.get(),
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
            .image = mSwapchainImages[imageIndex],
            .subresourceRange = {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1
            }
        };

        VkImageMemoryBarrier barriers[] = {dataBarrier, presentBarrier};
        vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr,
                             0, nullptr, 2, barriers);

        VkImageBlit blit{};
        blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        blit.srcSubresource.layerCount = 1;

        blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        blit.dstSubresource.layerCount = 1;

        const uint32_t width  = mSwapchain.extent().width;
        const uint32_t height = mSwapchain.extent().height;
        blit.srcOffsets[0] = {0, 0, 0};
        blit.srcOffsets[1] = {static_cast<int32_t>(width), static_cast<int32_t>(height), 1};

        blit.dstOffsets[0] = {0, 0, 0};
        blit.dstOffsets[1] = {static_cast<int32_t>(width), static_cast<int32_t>(height), 1};

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

        VkImageMemoryBarrier presentBarrier1{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask = 0,
            .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            .newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
            .image = mSwapchainImages[imageIndex],
            .subresourceRange = {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .levelCount = 1,
                .layerCount = 1
            }
        };
        VkPipelineStageFlagBits dstStage = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
        if (mRenderImGui) {
            presentBarrier1.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            presentBarrier1.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            dstStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        }

        vkCmdPipelineBarrier(
            commandBuffer,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            dstStage,
            0,
            0, nullptr,
            0, nullptr,
            1, &presentBarrier1
        );

        if (mRenderImGui) {
            ImGuiRenderInfo renderInfo {
                .commandBuffer = commandBuffer,
                .imageView = mSwapchainImageViews[imageIndex].get(),
                .extent = mSwapchain.extent()
            };
            mImGui.render(renderInfo);
            presentBarrier1.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            presentBarrier1.dstAccessMask = 0;
            presentBarrier1.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            presentBarrier1.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

            vkCmdPipelineBarrier(
                commandBuffer,
                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                0,
                0, nullptr,
                0, nullptr,
                1, &presentBarrier1
            );
        }
        
        if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
            throw std::runtime_error("Failed to record command buffer!");
        }
        vkResetFences(mContext.device(), 1, &mFences[mCurrentFrame].get());
    }

    void PathTracerApp::submit(const uint32_t imageIndex) {
        auto& commandBuffer = mGraphicsCommandBuffers[mCurrentFrame];
        VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
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

        VkQueue queue = mContext.queue(QueueFamilyType::GRAPHICS);
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

    void PathTracerApp::createRasterizer() {
        const auto [width, height] = mSwapchain.extent();
        const RasterizerCreateInfo createInfo {
            .context = &mContext,
            .colorFormat = VK_FORMAT_R8G8B8A8_UNORM,
            .extent = {width, height, 1},
            .vertices = mVertices,
            .indices = mIndices,
            .textures = &mTextures
        };
        mRasterizer = Rasterizer(createInfo);
    }
}