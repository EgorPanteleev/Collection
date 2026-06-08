//
// Created by igor on 6/7/26.
//

#include "PathTracerApp.hpp"
#include "CoreUtils.hpp"
#include "Timer.hpp"
#include "Loader.hpp"
#include "CallBacks.hpp"

#include <fstream>
#include <filesystem>

namespace fs = std::filesystem;

static glm::vec3 toVec3(const nlohmann::json& json) {
    return {
        json[0].get<float>(),
        json[1].get<float>(),
        json[2].get<float>()
    };
}

static VkTransformMatrixKHR toVkTransform(const glm::mat4& mat) {
    VkTransformMatrixKHR t{};
    for (int c = 0; c < 3; ++c)
        for (int r = 0; r < 4; ++r)
            t.matrix[c][r] = mat[r][c];
    return t;
}

namespace crv::graphics::vulkan {
    namespace cu = utils;

    PathTracerApp::PathTracerApp(const PathTracerAppCreateInfo& createInfo) {
        readScene(createInfo.scenePath);
        createContext();
        setCallBacks(this);
        createSwapChain();
        createSwapChainImages();
        createImages();
        createSyncObjects();
        createCommandBuffers();
        createCamera();
        loadScene();
        createRayTracerPass();
    }

    void PathTracerApp::run() {
        cu::FpsCounter fpsCounter;
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
            //++mFrameCount;
        }
        vkDeviceWaitIdle(mContext.device());
    }

    void PathTracerApp::readScene(const std::string& scenePath) {
        std::ifstream file(scenePath);
        file >> mScene;
    }

    void PathTracerApp::createContext() {
        auto window = mScene["window"];
        const WindowCreateInfo windowCreateInfo {
            .width  = window["width"],
            .height = window["height"],
            .name   = window["name"]
        };

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
                                     VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME},
            .enableValidationLayers = mDebug,
            .enableRT               = true
        };
        mContext = Context(createInfo);
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

    void PathTracerApp::createImages() {
        const ImageCreateInfo imageCreateInfo {
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
        mTracerImage = Image(imageCreateInfo);

        const ImageViewCreateInfo imageViewCreateInfo {
            .device = mContext.device(),
            .image = mTracerImage.get(),
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format = VK_FORMAT_R16G16B16A16_SFLOAT,
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
        };
        mTracerView = ImageView(imageViewCreateInfo);

        auto [commandBuffer, cmdData] = beginCommandBuffer(mContext.device(),
                                            mContext.familyIndex(QueueFamilyType::GRAPHICS).value());
        const ImageTransitInfo2 transitInfo {
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
        Image::transit(transitInfo);
        endCommandBuffer(cmdData, mContext.queue(QueueFamilyType::GRAPHICS));
    }

    void PathTracerApp::createSwapChainImages() {
        auto [capabilities, formats, presentModes] = Swapchain::getSupport(mContext.physicalDevice(), mContext.surface());
        uint32_t imageCount = Swapchain::getImageCount(capabilities);
        vkGetSwapchainImagesKHR(mContext.device(), mSwapchain.get(), &imageCount, nullptr);
        mSwapchainImages.resize(imageCount);
        vkGetSwapchainImagesKHR(mContext.device(), mSwapchain.get(), &imageCount, mSwapchainImages.data());

        auto [commandBuffer, cmdData] = beginCommandBuffer(mContext.device(), mContext.familyIndex(QueueFamilyType::COMPUTE).value());
        for (uint32_t i = 0; i < imageCount; ++i) {
            const ImageViewCreateInfo imageViewCreateInfo{
                .device = mContext.device(),
                .image = mSwapchainImages[i],
                .viewType = VK_IMAGE_VIEW_TYPE_2D,
                .format = mSwapchain.format(),
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
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
        endCommandBuffer(cmdData, mContext.queue(QueueFamilyType::COMPUTE));
    }

    void PathTracerApp::createSyncObjects() {
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
        for (uint32_t i = 0; i < mFramesInFlight; ++i) {
            mFences[i] = Fence(fenceCreateInfo);
            mImageAvailableSemaphores[i] = Semaphore(semaphoreCreateInfo);
        }
        mTracerFinishedSemaphores.resize(imageCount);
        for (uint32_t i = 0; i < imageCount; ++i) {
            mTracerFinishedSemaphores[i] = Semaphore(semaphoreCreateInfo);
        }
    }

    void PathTracerApp::createCommandBuffers() {
        const CommandPoolCreateInfo poolCreateInfo {
            .device = mContext.device(),
            .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            .queueFamilyIndex = mContext.familyIndex(QueueFamilyType::GRAPHICS).value()
        };
        mTracerCommandPool = CommandPool(poolCreateInfo);

        CommandBuffersCreateInfo bufferCreateInfo {
            .device = mContext.device(),
            .bufferCount = mFramesInFlight
        };
        bufferCreateInfo.commandPool = mTracerCommandPool.get();
        mTracerCommandBuffers = CommandBuffers(bufferCreateInfo);
    }

    void PathTracerApp::createCamera() {
        auto camera = mScene["camera"];
        auto window = mScene["window"];
        const cs::CameraCreateInfo info {
            .type = camera["type"] == "Fly" ? cs::CameraType::FLY : cs::CameraType::ORBITAL,
            .pos = toVec3(camera["position"]),
            .target = toVec3(camera["target"]),
            .up = toVec3(camera["up"]),
            .zoom = camera["zoom"],
            .FOV = camera["fov"],
            .aspectRatio = static_cast<float>(window["width"]) / static_cast<float>(window["height"]),
            .nearPlane = camera["nearPlane"],
            .farPlane = camera["farPlane"]
        };
        mFlyCamera = scene::FlyCamera(info);
        mOrbitalCamera = scene::OrbitalCamera(info);
        if (info.type == scene::CameraType::FLY) {
            mCamera = &mFlyCamera;
        } else {
            mCamera = &mOrbitalCamera;
        }
    }

    void PathTracerApp::loadModel(const uint32_t modelIndex, const std::string& path) {
        utils::Timer timer;
        timer.start();
        auto loader = new cm::Loader;
        loader->setModel(ASSETS_PATH + path);
        loader->load(glm::mat4(1.0f));
        INFO << "Model (" << fs::path(path).filename().stem().string() << ") load time: " << timer.duration() / 1000 << " sec";
        timer.start();

        auto allInstances = mScene["instances"];
        decltype(allInstances) instances;
        for (const auto& instance: allInstances) {
            if (instance["modelIndex"] != modelIndex) continue;
            instances.push_back(instance);
        }
        auto [commandBuffer, cmdData] = beginCommandBuffer(mContext.device(), mContext.familyIndex(QueueFamilyType::GRAPHICS).value());
        for (size_t meshIndex = 0; meshIndex < loader->meshes().size(); ++meshIndex) {
            const auto& mesh = loader->meshes()[meshIndex];
            std::vector<Vertex> vertices{};
            vertices.reserve(mesh.numVertices);
            for (size_t i = 0; i < mesh.numVertices; ++i) {
                const cm::Vertex& modelVertex = loader->vertices()[mesh.baseVertex + i];
                Vertex vertex {
                    .pos = modelVertex.pos,
                    .texCoord = modelVertex.texCoord0,
                    .normal = modelVertex.normal,
                    .tangent = modelVertex.tangent,
                };
                vertices.push_back(vertex);
            }
            std::vector<uint32_t> indices{};
            indices.reserve(mesh.numIndices);
            for (size_t i = 0; i < mesh.numIndices; ++i) {
                indices.push_back(loader->indices()[mesh.baseIndex + i]);
            }

            mBLASEntries.emplace_back();
            BLASEntry& blasEntry = mBLASEntries.back();
            const size_t verticesSize = sizeof(Vertex) * vertices.size();
            const BufferCreateInfo vertexBufferCreateInfo {
                .allocator = mContext.allocator(),
                .size = verticesSize,
                .bufferUsage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                               VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                .memoryUsage = VMA_MEMORY_USAGE_GPU_ONLY
            };
            blasEntry.vertexBuffer = Buffer(vertexBufferCreateInfo);
            const CopyDataToGPUBufferInfo vertexCopyInfo {
                .data = vertices.data(),
                .size = verticesSize,
                .allocator = mContext.allocator(),
                .buffer = blasEntry.vertexBuffer.get(),
                .device = mContext.device(),
                .queueFamilyIndex = mContext.familyIndex(QueueFamilyType::GRAPHICS).value(),
                .queue = mContext.queue(QueueFamilyType::GRAPHICS)
            };
            Buffer::copy(vertexCopyInfo);

            const size_t indicesSize = sizeof(uint32_t) * indices.size();
            const BufferCreateInfo indexBufferCreateInfo {
                .allocator = mContext.allocator(),
                .size = indicesSize,
                .bufferUsage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                               VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                .memoryUsage = VMA_MEMORY_USAGE_GPU_ONLY
            };
            blasEntry.indexBuffer = Buffer(indexBufferCreateInfo);
            const CopyDataToGPUBufferInfo indexCopyInfo {
                .data = indices.data(),
                .size = indicesSize,
                .allocator = mContext.allocator(),
                .buffer = blasEntry.indexBuffer.get(),
                .device = mContext.device(),
                .queueFamilyIndex = mContext.familyIndex(QueueFamilyType::GRAPHICS).value(),
                .queue = mContext.queue(QueueFamilyType::GRAPHICS)
            };
            Buffer::copy(indexCopyInfo);

            BLASCreateInfo blasCreateInfo {
                .commandBuffer = commandBuffer,
                .device = mContext.device(),
                .physicalDevice = mContext.physicalDevice(),
                .allocator = mContext.allocator(),
                .vertexAddress = blasEntry.vertexBuffer.deviceAddress(mContext.device()),
                .vertexStride = sizeof(Vertex),
                .vertexCount = static_cast<uint32_t>(vertices.size()),
                .indexAddress = blasEntry.indexBuffer.deviceAddress(mContext.device()),
                .indexCount = static_cast<uint32_t>(indices.size())
            };
            blasEntry.blas = AccelerationStructure(blasCreateInfo);
            VkDeviceAddress blasAddress = blasEntry.blas.deviceAddress();

            for (const auto& instance: instances) {
                glm::vec3 rot = toVec3(instance["localRotation"]);
                Transform transform;
                transform.position = toVec3(instance["localPosition"]);
                transform.scale = toVec3(instance["localScale"]);
                glm::quat qx = glm::angleAxis(glm::radians(rot.x), glm::vec3(1,0,0));
                glm::quat qy = glm::angleAxis(glm::radians(rot.y), glm::vec3(0,1,0));
                glm::quat qz = glm::angleAxis(glm::radians(rot.z), glm::vec3(0,0,1));
                transform.rotation = glm::normalize(qy * qx * qz);
                // uint32_t texIndex = instance["texIndex"];
                // if (texIndex == UINT32_MAX) texIndex = (baseMaterial + mesh.materialIndex) * cm::Texture::UNKNOWN;
                // else texIndex *= cm::Texture::UNKNOWN;
                VkASInstance asInstance{
                    .transform = toVkTransform(transform.matrix()),
                    .instanceCustomIndex = 0,
                    .mask = 0xFF,
                    .instanceShaderBindingTableRecordOffset = 0,
                    .flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR,
                    .accelerationStructureReference = blasAddress
                };
                mInstances.push_back(asInstance);
            }
        }
        endCommandBuffer(cmdData, mContext.queue(QueueFamilyType::GRAPHICS));
    }

    void PathTracerApp::loadScene() {
        // std::vector<std::string> textures = mScene["textureImports"];
        // auto materials = mScene["materials"];
        // mMaterials.resize(textures.size() + materials.size());
        // for (int textureIndex = 0; textureIndex < textures.size(); ++textureIndex) {
        //     mMaterials[textureIndex].mTextures[cm::Texture::DIFFUSE] =
        //         cm::AbsLoader::loadTexture(ASSETS_PATH + textures[textureIndex], cm::Texture::DIFFUSE);
        //     for (int texType = 1; texType < cm::Texture::UNKNOWN; ++texType) {
        //         mMaterials[textureIndex].mTextures[texType] =
        //             cm::AbsLoader::emptyTexture(static_cast<cm::Texture::Type>(texType));
        //     }
        // }
        //
        // int baseMaterial = static_cast<int>(textures.size());
        // for (int materialIndex = baseMaterial; materialIndex < baseMaterial + materials.size(); ++materialIndex) {
        //     mMaterials[materialIndex].mTextures[cm::Texture::DIFFUSE] =
        //         cm::AbsLoader::colorTexture(toVec3(materials[materialIndex]["color"]), cm::Texture::DIFFUSE);
        //     for (int texType = 1; texType < cm::Texture::UNKNOWN; ++texType) {
        //         mMaterials[materialIndex].mTextures[texType] =
        //             cm::AbsLoader::emptyTexture(static_cast<cm::Texture::Type>(texType));
        //     }
        // }

        std::vector<std::string> models = mScene["modelImports"];
          for (int modelIndex = 0; modelIndex < models.size(); ++modelIndex) {
            loadModel(modelIndex, models[modelIndex]);
        }

        const size_t instancesSize = sizeof(VkASInstance) * mInstances.size();
        const BufferCreateInfo instanceBufferCreateInfo {
            .allocator = mContext.allocator(),
            .size = instancesSize,
            .bufferUsage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                           VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .memoryUsage = VMA_MEMORY_USAGE_GPU_ONLY
        };
        mInstanceBuffer = Buffer(instanceBufferCreateInfo);
        const CopyDataToGPUBufferInfo instanceCopyInfo {
            .data = mInstances.data(),
            .size = instancesSize,
            .allocator = mContext.allocator(),
            .buffer = mInstanceBuffer.get(),
            .device = mContext.device(),
            .queueFamilyIndex = mContext.familyIndex(QueueFamilyType::GRAPHICS).value(),
            .queue = mContext.queue(QueueFamilyType::GRAPHICS)
        };
        Buffer::copy(instanceCopyInfo);

        auto [commandBuffer, cmdData] = beginCommandBuffer(mContext.device(), mContext.familyIndex(QueueFamilyType::GRAPHICS).value());
        TLASCreateInfo tlasCreateInfo {
            .commandBuffer = commandBuffer,
            .device = mContext.device(),
            .physicalDevice = mContext.physicalDevice(),
            .allocator = mContext.allocator(),
            .instanceAddress = mInstanceBuffer.deviceAddress(mContext.device()),
            .instanceCount = static_cast<uint32_t>(mInstances.size())
        };
        mTLAS = AccelerationStructure(tlasCreateInfo);
        endCommandBuffer(cmdData, mContext.queue(QueueFamilyType::GRAPHICS));
    }

    void PathTracerApp::createRayTracerPass() {
        const RayTracerPassCreateInfo createInfo {
            .context = &mContext,
            .tlas = &mTLAS,
            .outView = &mTracerView,
            .framesInFlight = mFramesInFlight
        };

        mRayTracerPass = RayTracerPass(createInfo);
    }

    void PathTracerApp::recordTracer(const uint32_t imageIndex) {
        VkCommandBuffer commandBuffer = mTracerCommandBuffers[mCurrentFrame];
        vkResetCommandBuffer(commandBuffer, 0);
        beginCommandBuffer(commandBuffer);
        const RayTracerPassRecordInfo recordInfo {
            .commandBuffer = commandBuffer,
            .width = mSwapchain.extent().width,
            .height = mSwapchain.extent().height
        };
        mRayTracerPass.record(recordInfo);
        recordPresent(imageIndex, commandBuffer);
        endCommandBuffer(commandBuffer);
    }

    void PathTracerApp::recordPresent(uint32_t imageIndex, VkCommandBuffer commandBuffer) {
        const ImageTransitInfo2 presentTransitInfo {
            .commandBuffer = commandBuffer,
            .image = mTracerImage.get(),
            .srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT,
            .srcStage = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
            .dstStage = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
            .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
        };

        const ImageTransitInfo2 swapchainTransitInfo {
            .commandBuffer = commandBuffer,
            .image = mSwapchainImages[imageIndex],
            .srcAccessMask = 0,
            .dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
            .srcStage = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            .dstStage = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
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
            mTracerImage.get(),
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            mSwapchainImages[imageIndex],
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1,
            &blit,
            VK_FILTER_NEAREST
        );

        Image::inverseTransit({presentTransitInfo, swapchainTransitInfo});
        // if (mRenderImGui) {
        //     inversePresentBarrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        //     inversePresentBarrier.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        //     dstStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        // }


        // if (mRenderImGui) {
        //     ImGuiRenderInfo renderInfo {
        //         .commandBuffer = commandBuffer,
        //         .imageView = mSwapchainImageViews[imageIndex].get(),
        //         .extent = mSwapchain.extent()
        //     };
        //     mImGui.render(renderInfo);
        //     inversePresentBarrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        //     inversePresentBarrier.dstAccessMask = 0;
        //     inversePresentBarrier.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        //     inversePresentBarrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        //
        //     ImagePipelineBarrierInfo guiBarrierInfo {
        //         .commandBuffer = commandBuffer,
        //         .srcStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        //         .dstStage = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
        //         .barriers = {inversePresentBarrier}
        //     };
        //     Image::pipelineBarrier(guiBarrierInfo);
        // }
    }

    void PathTracerApp::update() {
        const RayTracerPassUpdateInfo updateInfo {
            .camera = mCamera,
            .currentFrame = mCurrentFrame
        };
        mRayTracerPass.update(updateInfo);
    }

    void PathTracerApp::record(const uint32_t imageIndex) {
        recordTracer(imageIndex);
        vkResetFences(mContext.device(), 1, &mFences[mCurrentFrame].get());
    }

    void PathTracerApp::submit(const uint32_t imageIndex) {
        VkPipelineStageFlags tracerWaitStages[] = {VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR};
        const VkSubmitInfo tracerSubmitInfo{
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &mImageAvailableSemaphores[mCurrentFrame].get(),
            .pWaitDstStageMask = tracerWaitStages,
            .commandBufferCount = 1,
            .pCommandBuffers = &mTracerCommandBuffers[mCurrentFrame],
            .signalSemaphoreCount = 1,
            .pSignalSemaphores = &mTracerFinishedSemaphores[imageIndex].get()
        };
        VkQueue queue = mContext.queue(QueueFamilyType::GRAPHICS);
        if (vkQueueSubmit(queue, 1, &tracerSubmitInfo, mFences[mCurrentFrame].get()) != VK_SUCCESS) {
            throw std::runtime_error("Failed to submit draw command buffer!");
        }

        const VkPresentInfoKHR presentInfo {
            .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &mTracerFinishedSemaphores[imageIndex].get(),
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
            .fence = VK_NULL_HANDLE,
            .imageIndex = &imageIndex
        };
        const VkResult result = mSwapchain.acquireNextImage(swapchainAcquireInfo);
        if (result != VK_SUCCESS) {
            throw std::runtime_error("Failed to acquire image!");
        }
    }

    void PathTracerApp::drawFrame() {
        uint32_t imageIndex;
        vkWaitForFences(mContext.device(), 1, &mFences[mCurrentFrame].get(), VK_TRUE, UINT64_MAX);
        acquireNextImage(imageIndex);
//        drawControlPanel();
        update();
        record(imageIndex);
        submit(imageIndex);
    }
}
