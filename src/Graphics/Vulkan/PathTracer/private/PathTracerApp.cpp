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

#include <chrono>
#include <fstream>
#include <filesystem>

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

    void PathTracerApp::pixelClicked(uint32_t x, uint32_t y) {
        auto [width, height] = mSwapchain.extent();
        if (x > width or y > height) return;
        mClickedPixel = {x, y};
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
            .resourceManager = &mResourceManager
        };
        mUI = AppUI(createInfo);

        mUI.setUpdateImageCallBack([this](){updateImage();});
        mUI.setCameraSetCallBack([this](cs::CameraType type){setCamera(type);});
        mUI.setUploadTextureCallBack([this](const std::string& path, uint32_t materialIndex, int textureType){
            vkDeviceWaitIdle(mContext.device());
            uint32_t index = 0;
            switch (textureType) {
                case 1:  index = mResourceManager.addNormalTexture(path, materialIndex); break;
                case 2:  index = mResourceManager.addMetalRoughnessTexture(path, materialIndex); break;
                default: index = mResourceManager.addBaseColorTexture(path, materialIndex); break;
            }
            mRayTracerPass.bindTexture(index);
        });
        mUI.setSaveImageCallBack([this](){ saveImage(); });
        mUI.setSaveSceneCallBack([this](){ saveScene(); });
    }

    void PathTracerApp::recordTracer() {
        VkCommandBuffer commandBuffer = mTracerCommandBuffers[mCurrentFrame];
        vkResetCommandBuffer(commandBuffer, 0);
        beginCommandBuffer(commandBuffer);
        const RayTracerPassRecordInfo recordInfo {
            .commandBuffer = commandBuffer,
            .constants = {
                .frameCount = mFrameCount,
                .spp = mUI.spp(),
                .minDepth = mUI.minDepth(),
                .maxDepth = mUI.maxDepth(),
                .displayMode = mUI.displayMode(),
                .nee = mUI.nee() ? 1u : 0u,
                .emissiveCount = static_cast<uint32_t>(mResourceManager.emissiveIndices().size())
            },
            .width = mSwapchain.extent().width,
            .height = mSwapchain.extent().height
        };
        mRayTracerPass.record(recordInfo);
        recordPixelRead(commandBuffer);
        endCommandBuffer(commandBuffer);
    }

    void PathTracerApp::recordRaster() {
        VkCommandBuffer commandBuffer = mRasterCommandBuffers[mCurrentFrame];
        vkResetCommandBuffer(commandBuffer, 0);
        beginCommandBuffer(commandBuffer);
        RasterizerPassRecordInfo recordInfo{};
        if (mSelectedInstanceId != 0) {
            const InstanceData& instance = mResourceManager.instances()[mSelectedInstanceId - 1];
            BLASData& blasData = mResourceManager.blasDatas()[instance.meshIndex];
            recordInfo = {
                .commandBuffer = commandBuffer,
                .vertexBuffer = &blasData.vertexBuffer,
                .indexBuffer = &blasData.indexBuffer,
                .indexCount = instance.indexCount,
                .extent = mSwapchain.extent(),
                .currentFrame = mCurrentFrame
            };
        } else {
            recordInfo = {
                .commandBuffer = commandBuffer,
                .vertexBuffer = nullptr,
                .indexBuffer = nullptr,
                .indexCount = 0,
                .extent = mSwapchain.extent(),
                .currentFrame = mCurrentFrame
            };
        }
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
                .exposure = mUI.exposure(),
                .tonemap = mUI.tonemap() ? 1u : 0u,
                .displayMode = mUI.displayMode()
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

        const VkBufferImageCopy region {
            .bufferOffset = 0,
            .bufferRowLength = 0,
            .bufferImageHeight = 0,
            .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
            .imageOffset = {static_cast<int32_t>(mClickedPixel.x), static_cast<int32_t>(mClickedPixel.y), 0},
            .imageExtent = {1, 1, 1}
        };
        vkCmdCopyImageToBuffer(commandBuffer, mTracerInstanceImage.get(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            mReadbackBuffer.get(),
            1, &region);

        Image::inverseTransit(transitInfo);
        mClickedPixel = {UINT32_MAX, UINT32_MAX};
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
        uint32_t* data = nullptr;
        vmaMapMemory(mContext.allocator(), mReadbackBuffer.allocation(), (void**)&data);
        mSelectedInstanceId = *data;
        vmaUnmapMemory(mContext.allocator(), mReadbackBuffer.allocation());
    }

    void PathTracerApp::clearSelection() {
        uint32_t* data = nullptr;
        vmaMapMemory(mContext.allocator(), mReadbackBuffer.allocation(), (void**)&data);
        *data = 0;
        vmaUnmapMemory(mContext.allocator(), mReadbackBuffer.allocation());
        mSelectedInstanceId = 0;
    }

    void PathTracerApp::update() {
        const RayTracerPassUpdateInfo tracerUpdateInfo {
            .camera = mCamera,
            .directLight = mResourceManager.directLight(),
            .currentFrame = mCurrentFrame
        };
        mRayTracerPass.update(tracerUpdateInfo);

        if (mSelectedInstanceId == 0) return;
        const RasterizerPassUpdateInfo rasterUpdateInfo {
            .camera = mCamera,
            .instance = &mResourceManager.instances()[mSelectedInstanceId - 1],
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
            .selectedInstanceIndex = mSelectedInstanceId,
            .frameCount = mFrameCount
        };
        mUI.draw(drawInfo);
    }

    void PathTracerApp::drawFrame() {
        uint32_t imageIndex;
        vkWaitForFences(mContext.device(), 1, &mFences[mCurrentFrame].get(), VK_TRUE, UINT64_MAX);
        updateSelectedInstance();
        acquireNextImage(imageIndex);
        drawControlPanel();
        update();
        record(imageIndex);
        submit(imageIndex);
    }
}
