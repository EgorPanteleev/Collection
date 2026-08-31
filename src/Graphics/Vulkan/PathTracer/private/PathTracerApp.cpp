//
// Created by igor on 6/7/26.
//

#include "PathTracerApp.hpp"
#include "CoreUtils.hpp"
#include "Timer.hpp"
#include "Loader.hpp"
#include "CallBacks.hpp"
#include "IconsFontAwesome6.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <filesystem>
#include <limits>

namespace fs = std::filesystem;

namespace crv::graphics::vulkan {
    namespace cu = utils;

    namespace {
        void dumpScene(std::ostream& out, const json& j, int indent, int depth) {
            const std::string pad(static_cast<size_t>(depth) * indent, ' ');
            const std::string padIn(static_cast<size_t>(depth + 1) * indent, ' ');
            if (j.is_object()) {
                if (j.empty()) { out << "{}"; return; }
                out << "{\n";
                size_t i = 0;
                for (auto it = j.begin(); it != j.end(); ++it) {
                    out << padIn << json(it.key()).dump() << ": ";
                    dumpScene(out, it.value(), indent, depth + 1);
                    out << (++i < j.size() ? ",\n" : "\n");
                }
                out << pad << "}";
            } else if (j.is_array()) {
                bool inlineArray = true;
                for (const auto& e : j) if (!e.is_number() && !e.is_boolean()) { inlineArray = false; break; }
                if (inlineArray) {
                    out << "[";
                    size_t i = 0;
                    for (const auto& e : j) out << (i++ ? ", " : "") << e.dump();
                    out << "]";
                } else {
                    out << "[\n";
                    size_t i = 0;
                    for (const auto& e : j) {
                        out << padIn;
                        dumpScene(out, e, indent, depth + 1);
                        out << (++i < j.size() ? ",\n" : "\n");
                    }
                    out << pad << "]";
                }
            } else {
                out << j.dump();
            }
        }
    }

    PathTracerApp::PathTracerApp(const PathTracerAppCreateInfo& createInfo) {
        readScene(createInfo.scenePath);
        createContext();
        setCallBacks(this);
        createSwapChain();
        createSwapChainImages();
        createBuffers();
        createImages();
        createSyncObjects();
        createCommandBuffers();
        createCamera();
        createResourceManager();
        loadScene();
        createRayTracerPass();
        createRasterizerPass();
        createPostprocessPass();
        createUI();
    }

    void PathTracerApp::run() {
        cu::FpsCounter fpsCounter;
        double deltaTime = 0;
        const Window& window = mContext.window();
        while (!window.shouldClose()) {
            mInput.beginFrame();
            glfwPollEvents();
            window.keyboardCallBack(deltaTime);
            applyCommands(deltaTime);
            fpsCounter.update();
            deltaTime = 1e3 / fpsCounter.fps();
            window.setTitle(std::to_string(fpsCounter.fps()).c_str());
            drawFrame();
            updateCurrentFrame();
            ++mFrameCount;
        }
        vkDeviceWaitIdle(mContext.device());
    }

    void PathTracerApp::pixelClicked(uint32_t x, uint32_t y, bool additive) {
        auto [width, height] = mSwapchain.extent();
        if (x > width or y > height) return;
        mClickedPixel = {x, y};
        mAdditiveSelect = additive;
    }

    void PathTracerApp::readScene(const std::string& scenePath) {
        std::ifstream file(scenePath);
        file >> mJson;
    }

    void PathTracerApp::createContext() {
        auto window = mJson["window"];
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

    void PathTracerApp::createBuffers() {
        const BufferCreateInfo readbackInfo {
            .allocator = mContext.allocator(),
            .size = sizeof(uint32_t),
            .bufferUsage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .memoryUsage = VMA_MEMORY_USAGE_CPU_ONLY
        };
        mReadbackBuffer = Buffer(readbackInfo);
    }

    void PathTracerApp::createImages() {
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
        mTracerImage = Image(imageCreateInfo);
        mFinalImage  = Image(imageCreateInfo);

        imageCreateInfo.format = VK_FORMAT_R32_UINT;
        mTracerInstanceImage = Image(imageCreateInfo);
        imageCreateInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
        mRasterInstanceImage = Image(imageCreateInfo);

        ImageViewCreateInfo imageViewCreateInfo {
            .device = mContext.device(),
            .image = mTracerImage.get(),
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format = VK_FORMAT_R16G16B16A16_SFLOAT,
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
        };
        mTracerView = ImageView(imageViewCreateInfo);
        imageViewCreateInfo.image = mFinalImage.get();
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

    void PathTracerApp::createCommandBuffers() {
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

    void PathTracerApp::createCamera() {
        auto camera = mJson["camera"];
        auto window = mJson["window"];
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

    void PathTracerApp::createResourceManager() {
        const ResourceManagerCreateInfo createInfo {
            .context = &mContext
        };
        mResourceManager = ResourceManager(createInfo);
    }

    void PathTracerApp::loadScene() {
        mResourceManager.load(mJson);
    }

    void PathTracerApp::createRayTracerPass() {
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

    void PathTracerApp::createRasterizerPass() {
        const RasterizerPassCreateInfo createInfo {
            .context = &mContext,
            .outView = &mRasterInstanceView,
            .outFormat = VK_FORMAT_R32_UINT,
            .extent = mSwapchain.extent(),
            .framesInFlight = mFramesInFlight
        };
        mRasterizerPass = RasterizerPass(createInfo);
    }

    void PathTracerApp::createPostprocessPass() {
        const PostprocessPassCreateInfo createInfo {
            .context = &mContext,
            .tracerView = &mTracerView,
            .instanceView = &mRasterInstanceView,
            .outputView = &mFinalView,
            .framesInFlight = mFramesInFlight
        };
        mPostprocessPass = PostprocessPass(createInfo);
    }

    void PathTracerApp::createUI() {
        const AppUICreateInfo createInfo {
            .context = &mContext,
            .swapchain = &mSwapchain,
            .resourceManager = &mResourceManager,
            .renderSettings = &mRenderSettings,
            .commands = &mCommands
        };
        mUI = AppUI(createInfo);
    }

    void PathTracerApp::selectInstance(const uint32_t index, const bool additive) {
        applySelection(index + 1, additive);
    }

    void PathTracerApp::uploadTexture(const std::string& path, const uint32_t materialIndex, const int textureType) {
        vkDeviceWaitIdle(mContext.device());
        uint32_t index = 0;
        switch (textureType) {
            case 1:  index = mResourceManager.addNormalTexture(path, materialIndex); break;
            case 2:  index = mResourceManager.addMetalRoughnessTexture(path, materialIndex); break;
            case 3:  index = mResourceManager.addClearcoatTexture(path, materialIndex); break;
            case 4:  index = mResourceManager.addClearcoatRoughnessTexture(path, materialIndex); break;
            default: index = mResourceManager.addBaseColorTexture(path, materialIndex); break;
        }
        mRayTracerPass.bindTexture(index);
    }

    void PathTracerApp::loadSkybox(const std::string& path) {
        vkDeviceWaitIdle(mContext.device());
        const uint32_t index = mResourceManager.addSkybox(path);
        mRayTracerPass.bindTexture(index);
        updateImage();
    }

    void PathTracerApp::removeSkybox() {
        mResourceManager.removeSkybox();
        updateImage();
    }

    void PathTracerApp::duplicateInstances(const std::vector<uint32_t>& indices) {
        vkDeviceWaitIdle(mContext.device());
        const std::vector<uint32_t> created = mResourceManager.duplicateInstances(indices);
        mRayTracerPass.bindInstances();
        mSelectedInstances = created;
        mActiveInstance = created.empty() ? UINT32_MAX : created.back();
        mPendingSelection = false;
        updateImage();
    }

    void PathTracerApp::removeInstances(const std::vector<uint32_t>& indices) {
        vkDeviceWaitIdle(mContext.device());
        mResourceManager.removeInstances(indices);
        mRayTracerPass.bindInstances();
        clearSelection();
        updateImage();
    }

    void PathTracerApp::addMaterial(const uint32_t instanceIndex) {
        vkDeviceWaitIdle(mContext.device());
        auto& instances = mResourceManager.instances();
        if (instanceIndex >= instances.size()) return;
        Material newMaterial = mResourceManager.materials()[instances[instanceIndex].materialIndex];
        newMaterial.name += " copy";
        const uint32_t index = mResourceManager.addMaterial(newMaterial);
        instances[instanceIndex].materialIndex = index;
        mResourceManager.updateInstance(instanceIndex);
        mRayTracerPass.bindMaterials();
        updateImage();
    }

    void PathTracerApp::recordTracer() {
        VkCommandBuffer commandBuffer = mTracerCommandBuffers[mCurrentFrame];
        vkResetCommandBuffer(commandBuffer, 0);
        beginCommandBuffer(commandBuffer);
        const RayTracerPassRecordInfo recordInfo {
            .commandBuffer = commandBuffer,
            .constants = {
                .frameCount = mFrameCount,
                .spp = static_cast<uint32_t>(mRenderSettings.spp),
                .minDepth = static_cast<uint32_t>(mRenderSettings.minDepth),
                .maxDepth = static_cast<uint32_t>(mRenderSettings.maxDepth),
                .displayMode = static_cast<uint32_t>(mRenderSettings.displayMode),
                .nee = mRenderSettings.nee ? 1u : 0u,
                .emissiveCount = static_cast<uint32_t>(mResourceManager.emissiveIndices().size()),
                .skyboxIndex = mResourceManager.skyboxIndex(),
                .envIntegral = mResourceManager.envIntegral(),
                .envNee = mRenderSettings.envNee ? 1u : 0u,
                .aperture = mRenderSettings.aperture,
                .focusDistance = mRenderSettings.focusDistance,
                .envMarginalCdfAddr = mResourceManager.envMarginalCdfAddr(),
                .envCondCdfAddr = mResourceManager.envCondCdfAddr(),
                .envCondFuncAddr = mResourceManager.envCondFuncAddr(),
                .skyColor = mResourceManager.skyColor()
            },
            .width = (mSwapchain.extent().width + mEffectiveScale - 1) / mEffectiveScale,
            .height = (mSwapchain.extent().height + mEffectiveScale - 1) / mEffectiveScale
        };
        mRayTracerPass.record(recordInfo);
        recordPixelRead(commandBuffer);
        endCommandBuffer(commandBuffer);
    }

    void PathTracerApp::recordRaster() {
        VkCommandBuffer commandBuffer = mRasterCommandBuffers[mCurrentFrame];
        vkResetCommandBuffer(commandBuffer, 0);
        beginCommandBuffer(commandBuffer);
        std::vector<RasterizerDraw> draws;
        draws.reserve(mSelectedInstances.size());
        uint32_t outlineId = 1;
        for (const uint32_t index : mSelectedInstances) {
            const InstanceData& instance = mResourceManager.instances()[index];
            BLASData& blasData = mResourceManager.blasDatas()[instance.meshIndex];
            draws.push_back({
                .vertexBuffer = &blasData.vertexBuffer,
                .indexBuffer = &blasData.indexBuffer,
                .indexCount = instance.indexCount,
                .model = instance.transform.matrix(),
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

    void PathTracerApp::recordPostprocess() {
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
                .exposure = mRenderSettings.exposure,
                .tonemap = mRenderSettings.tonemap ? 1u : 0u,
                .displayMode = static_cast<uint32_t>(mRenderSettings.displayMode),
                .renderScale = mEffectiveScale
            }
        };
        mPostprocessPass.record(recordInfo);
        Image::inverseTransit(transitInfo);
    }

    void PathTracerApp::recordPresent(uint32_t imageIndex) {
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

    void PathTracerApp::recordPixelRead(VkCommandBuffer commandBuffer) {
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
        mPendingSelection = true;
    }

    void PathTracerApp::regionSelect(int x0, int y0, int x1, int y1, bool additive) {
        auto [width, height] = mSwapchain.extent();
        const float fw = static_cast<float>(width);
        const float fh = static_cast<float>(height);
        const float nx0 = static_cast<float>(std::min(x0, x1)) / fw * 2.0f - 1.0f;
        const float nx1 = static_cast<float>(std::max(x0, x1)) / fw * 2.0f - 1.0f;
        const float ny0 = static_cast<float>(std::min(y0, y1)) / fh * 2.0f - 1.0f;
        const float ny1 = static_cast<float>(std::max(y0, y1)) / fh * 2.0f - 1.0f;

        const glm::mat4 viewProj = mCamera->projectionMatrix() * mCamera->viewMatrix();
        const auto& instances = mResourceManager.instances();
        const auto& blasDatas = mResourceManager.blasDatas();

        if (!additive) mSelectedInstances.clear();
        for (uint32_t i = 0; i < instances.size(); ++i) {
            const InstanceData& instance = instances[i];
            const BLASData& mesh = blasDatas[instance.meshIndex];
            const glm::mat4 mvp = viewProj * instance.transform.matrix();

            glm::vec2 boxMin(std::numeric_limits<float>::max());
            glm::vec2 boxMax(std::numeric_limits<float>::lowest());
            bool anyInFront = false;
            for (int c = 0; c < 8; ++c) {
                const glm::vec4 corner {
                    (c & 1) ? mesh.aabbMax.x : mesh.aabbMin.x,
                    (c & 2) ? mesh.aabbMax.y : mesh.aabbMin.y,
                    (c & 4) ? mesh.aabbMax.z : mesh.aabbMin.z,
                    1.0f
                };
                const glm::vec4 clip = mvp * corner;
                if (clip.w <= 1e-4f) continue;
                anyInFront = true;
                const glm::vec2 ndc = glm::vec2(clip) / clip.w;
                boxMin = glm::min(boxMin, ndc);
                boxMax = glm::max(boxMax, ndc);
            }
            if (!anyInFront) continue;
            if (boxMax.x < nx0 || boxMin.x > nx1 || boxMax.y < ny0 || boxMin.y > ny1) continue;

            if (std::find(mSelectedInstances.begin(), mSelectedInstances.end(), i) == mSelectedInstances.end())
                mSelectedInstances.push_back(i);
        }
        mActiveInstance = mSelectedInstances.empty() ? UINT32_MAX : mSelectedInstances.back();
    }

    void PathTracerApp::saveImage() {
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

    void PathTracerApp::saveScene() {
        const json scene = mResourceManager.saveScene();
        const auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        char stamp[32];
        std::strftime(stamp, sizeof(stamp), "%Y%m%d_%H%M%S", std::localtime(&now));
        const std::string path = (fs::path(ASSETS_PATH) / ("scene_" + std::string(stamp) + ".json")).string();
        std::ofstream out(path);
        if (!out) {
            ERROR << "Failed to save scene: " << path;
            return;
        }
        dumpScene(out, scene, 2, 0);
        out << "\n";
        INFO << "Saved scene: " << path;
    }

    void PathTracerApp::updateSelectedInstance() {
        if (mPendingSelection) {
            mPendingSelection = false;
            uint32_t* data = nullptr;
            vmaMapMemory(mContext.allocator(), mReadbackBuffer.allocation(), (void**)&data);
            const uint32_t id = *data;
            vmaUnmapMemory(mContext.allocator(), mReadbackBuffer.allocation());
            applySelection(id, mAdditiveSelect);
        }
    }

    void PathTracerApp::applySelection(const uint32_t id, const bool additive) {
        if (id == 0) {
            if (!additive) clearSelection();
            return;
        }
        const uint32_t index = id - 1;
        const auto it = std::find(mSelectedInstances.begin(), mSelectedInstances.end(), index);
        if (!additive) {
            mSelectedInstances = {index};
            mActiveInstance = index;
        } else if (it != mSelectedInstances.end()) {
            mSelectedInstances.erase(it);
            mActiveInstance = mSelectedInstances.empty() ? UINT32_MAX : mSelectedInstances.back();
        } else {
            mSelectedInstances.push_back(index);
            mActiveInstance = index;
        }
    }

    void PathTracerApp::clearSelection() {
        mSelectedInstances.clear();
        mActiveInstance = UINT32_MAX;
        mPendingSelection = false;
    }

    void PathTracerApp::update() {
        const RayTracerPassUpdateInfo tracerUpdateInfo {
            .camera = mCamera,
            .directLight = mResourceManager.directLight(),
            .currentFrame = mCurrentFrame
        };
        mRayTracerPass.update(tracerUpdateInfo);

        const RasterizerPassUpdateInfo rasterUpdateInfo {
            .camera = mCamera,
            .currentFrame = mCurrentFrame
        };
        mRasterizerPass.update(rasterUpdateInfo);
    }

    void PathTracerApp::record(const uint32_t imageIndex) {
        recordTracer();
        recordRaster();
        recordPostprocess();
        recordPresent(imageIndex);
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

    void PathTracerApp::setCamera(const scene::CameraType type) {
        if (type == scene::CameraType::FLY) {
            mCamera = &mFlyCamera;
            mCamera->setPosition(mOrbitalCamera.position());
            mCamera->setOrientation(mOrbitalCamera.orientation());
        } else {
            mCamera = &mOrbitalCamera;
        }
    }

    void PathTracerApp::drawControlPanel() {
        const AppUIDrawInfo drawInfo {
            .drawUI = mRenderImGui,
            .camera = mCamera,
            .selectedInstances = &mSelectedInstances,
            .activeInstance = mActiveInstance,
            .frameCount = mFrameCount,
            .renderScale = mEffectiveScale
        };
        mUI.draw(drawInfo);
    }

    void PathTracerApp::applyCommands(const double deltaTime) {
        bool cameraMoved = false;
        const CameraInput cameraInput {
            .camera    = mCamera,
            .input     = &mInput,
            .deltaTime = static_cast<float>(deltaTime),
        };
        for (const Command& command : mCommands.get()) {
            switch (commandTarget(command.type)) {
                case CommandTarget::CAMERA:
                    if (mCameraHandler.apply(command, cameraInput)) cameraMoved = true;
                    else mAppHandler.apply(command, this);
                    break;
                case CommandTarget::SCENE:
                case CommandTarget::VIEW:
                case CommandTarget::APP:
                    mAppHandler.apply(command, this);
                    break;
                default:
                    break;
            }
        }
        mCommands.clear();
        if (cameraMoved) onCameraMoved();
    }

    void PathTracerApp::pickAtCursor() {
        const glm::dvec2 cursor = mInput.cursorPos();
        GLFWwindow* window = mContext.window().glfwWindow();
        int winWidth, winHeight, fbWidth, fbHeight;
        glfwGetWindowSize(window, &winWidth, &winHeight);
        glfwGetFramebufferSize(window, &fbWidth, &fbHeight);
        const float scaleX = static_cast<float>(fbWidth)  / static_cast<float>(winWidth);
        const float scaleY = static_cast<float>(fbHeight) / static_cast<float>(winHeight);
        const bool additive = mInput.isPressed(Key::LEFT_SHIFT) || mInput.isPressed(Key::RIGHT_SHIFT);
        pixelClicked(static_cast<uint32_t>(cursor.x * scaleX),
                     static_cast<uint32_t>(cursor.y * scaleY), additive);
    }

    void PathTracerApp::drawFrame() {
        uint32_t imageIndex;
        vkWaitForFences(mContext.device(), 1, &mFences[mCurrentFrame].get(), VK_TRUE, UINT64_MAX);
        updateSelectedInstance();
        acquireNextImage(imageIndex);
        drawControlPanel();

        const uint32_t scale = mCameraMoved ? mRenderSettings.effectiveMotionScale() : mRenderSettings.effectiveRenderScale();
        if (scale != mEffectiveScale) {
            mEffectiveScale = scale;
            mFrameCount = 0;
        }

        update();
        record(imageIndex);
        submit(imageIndex);
        mCameraMoved = false;
    }
}
