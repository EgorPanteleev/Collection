//
// Created by igor on 6/7/26.
//

#include "PathTracerApp.hpp"
#include "CoreUtils.hpp"

#include <fstream>

static glm::vec3 toVec3(const nlohmann::json& json) {
    return {
        json[0].get<float>(),
        json[1].get<float>(),
        json[2].get<float>()
    };
}

namespace crv::graphics::vulkan {
    PathTracerApp::PathTracerApp(const PathTracerAppCreateInfo& createInfo) {
        readScene(createInfo.scenePath);
        createContext();
        createSwapChain();
        createImages();
        createRayTracerPass();
        createCamera();
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

    void PathTracerApp::createRayTracerPass() {
        const RayTracerPassCreateInfo createInfo {
            .context = &mContext,
            .outView = &mTracerView,
            .framesInFlight = mFramesInFlight
        };

        mRayTracerPass = RayTracerPass(createInfo);
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
        endCommandBuffer(cmdData, mContext.queue(QueueFamilyType::COMPUTE));
    }
}
