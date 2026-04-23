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
        mCamera = std::make_unique<scene::FlyCamera>(info.cameraCreateInfo);
        mDirectLight = info.directLight;
        createContext(info.windowCreateInfo);
        setCallBacks(this);
        createCommandPool();
        createCommandBuffers();
        createSwapChain();
        createSwapChainImages();
        createSyncObjects();
        createPresentImage();
        createPathTracer(info);
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
            PathTracerUpdateInfo pathTracerUpdateInfo {
                .camera = mCamera.get(),
                .directLight = mDirectLight,
                .presentImage = mPresentImage.get(),
                .presentImageView = mPresentImageView.get(),
                .currentFrame = mCurrentFrame
            };
            mPathTracer.update(pathTracerUpdateInfo);

            if (mRenderImGui) {
                mImGui.beginFrame();
                mImGui.demo();
                mImGui.endFrame();
            }

            record(imageIndex);
            submit(imageIndex);
            updateCurrentFrame();
        }
        vkDeviceWaitIdle(mContext.device());
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
        const CommandPoolCreateInfo createInfo {
            .device = mContext.device(),
            .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            .queueFamilyIndex = mContext.familyIndex(QueueFamilyType::COMPUTE).value()
        };
        mCommandPool = CommandPool(createInfo);
    }

    void PathTracerApp::createCommandBuffers() {
        const CommandBuffersCreateInfo createInfo {
            .device = mContext.device(),
            .commandPool = mCommandPool.get(),
            .bufferCount = mFramesInFlight
        };
        mCommandBuffers = CommandBuffers(createInfo);
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

    static BVH buildBVH(std::span<Tri> tris) {
        utils::Timer timer;
        timer.start();
        BinnedSAHBuilder<BVH::Node, Tri> builder{ tris };
        BVH bvh = builder.build();
        INFO << "BVH build time: " << timer.duration() / 1000 << " sec";
        return bvh;
    }

    static auto loadModel(const PathTracerAppCreateInfo& info) {
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
        std::vector<uint32_t> tmpIndices;
        std::vector<uint32_t> tmpMaterialIndices;
        for (size_t i = 0; i < lMeshes.size(); ++i) {
            const auto& mesh = lMeshes[i];
            for (size_t j = 0; j < mesh.numIndices; j += 3) {
                const size_t idx = mesh.baseIndex + j;
                triangles.emplace_back(lVertices[lIndices[idx + 0]].pos,
                                       lVertices[lIndices[idx + 1]].pos,
                                       lVertices[lIndices[idx + 2]].pos);
                tmpIndices.emplace_back(lIndices[idx + 0]);
                tmpIndices.emplace_back(lIndices[idx + 1]);
                tmpIndices.emplace_back(lIndices[idx + 2]);
                tmpMaterialIndices.emplace_back(mesh.materialIndex);
            }
        }
        INFO << "Primitive creation time: " << timer.duration() / 1000 << " sec";
        INFO << "Total number of primitives: " << triangles.size();
        auto bvh = buildBVH(std::span(triangles));
        //reorder after bvh
        std::vector<AlignedTriangle> alignedTriangles;
        std::vector<AlignedTriangleExtra> alignedTriangleExtras;
        std::vector<uint32_t> materialIndices;
        for (int i = 0; i < triangles.size(); ++i) {
            const uint32_t idx = bvh.primIds()[i];
            Tri& tri = triangles[idx];
            alignedTriangles.emplace_back(Vec4(tri.p0, 1), Vec4(tri.e1, 1),
                                          Vec4(tri.e2, 1), Vec4(tri.N, 1));
            cm::Vertex v0 = lVertices[tmpIndices[idx * 3 + 0]];
            cm::Vertex v1 = lVertices[tmpIndices[idx * 3 + 1]];
            cm::Vertex v2 = lVertices[tmpIndices[idx * 3 + 2]];
            alignedTriangleExtras.emplace_back(v0.texCoord0, v1.texCoord0, v2.texCoord0);
            materialIndices.push_back(tmpMaterialIndices[idx]);
        }

        std::vector<AlignedNode> nodes;
        for (const auto& node: bvh.nodes()) {
            AlignedNode alignedNode{};
            alignedNode.bbox = AlignedBBox(Vec4(node.bbox().min, 1), Vec4(node.bbox().max, 1));
            alignedNode.index = node.index().value();
            nodes.push_back(alignedNode);
        }
        return std::make_tuple(alignedTriangles, alignedTriangleExtras, nodes, loader->materials(), materialIndices);
    }

    void PathTracerApp::createPathTracer(const PathTracerAppCreateInfo& createInfo) {
        auto [triangles, triangleExtras, nodes, materials, materialIndices] =
            loadModel(createInfo);
        const PathTracerCreateInfo pathTracerCreateInfo {
            .context = &mContext,
            .triangles = triangles,
            .triangleExtras = triangleExtras,
            .nodes = nodes,
            .materials = materials,
            .materialIndices = materialIndices,
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

    void PathTracerApp::record(uint32_t imageIndex) {
        VkCommandBuffer commandBuffer = mCommandBuffers[mCurrentFrame]; 
        vkResetCommandBuffer(commandBuffer, 0);
        constexpr VkCommandBufferBeginInfo beginInfo {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = 0,
            .pInheritanceInfo = nullptr
        };
        if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
            throw std::runtime_error("Failed to begin recording command buffer!");
        }

        PathTracerRecordInfo pathTracerRecordInfo {
            .commandBuffer = commandBuffer,
            .extent = mSwapchain.extent(),
            .currentFrame = mCurrentFrame
        };
        mPathTracer.record(pathTracerRecordInfo);
        
        const VkImageMemoryBarrier dataBarrier{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
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
        vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
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
}