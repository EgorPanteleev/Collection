//
// Created by igor on 6/7/26.
//

#include "PathTracerApp.hpp"
#include "CoreUtils.hpp"
#include "Timer.hpp"
#include "Loader.hpp"
#include "CallBacks.hpp"
#include "IconsFontAwesome6.h"

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

static float clampAngle(float deg) {
    deg = std::fmod(deg + 180.0f, 360.0f);
    if (deg < 0.0f) deg += 360.0f;
    return deg - 180.0f;
}


static glm::vec3 clampRotation(const glm::vec3& e) {
    return {clampAngle(e.x), clampAngle(e.y), clampAngle(e.z)};
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
        createBuffers();
        createImages();
        createSyncObjects();
        createCommandBuffers();
        createCamera();
        loadScene();
        createTextures();
        createRayTracerPass();
        createRasterizerPass();
        createPostprocessPass();
        createImGui();
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
        if (!mRenderImGui) return;
        auto [width, height] = mSwapchain.extent();
        if (x > width or y > height) return;
        mClickedPixel = {x, y};
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
                               VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                               VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
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
                               VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                               VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
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
            mBLASInfos.emplace_back(blasEntry.vertexBuffer.deviceAddress(mContext.device()),
                blasEntry.indexBuffer.deviceAddress(mContext.device()));
            VkDeviceAddress blasAddress = blasEntry.blas.deviceAddress();
            auto allInstances = mScene["instances"];
            decltype(allInstances) instances;
            for (const auto& instance: allInstances) {
                if (instance["modelIndex"] != modelIndex) continue;
                instances.push_back(instance);
            }

            uint32_t baseMaterial = mMaterials.size();
            for (const auto& instance: instances) {
                glm::vec3 rot = toVec3(instance["localRotation"]);
                Transform transform;
                transform.position = toVec3(instance["localPosition"]);
                transform.scale = toVec3(instance["localScale"]);
                glm::quat qx = glm::angleAxis(glm::radians(rot.x), glm::vec3(1,0,0));
                glm::quat qy = glm::angleAxis(glm::radians(rot.y), glm::vec3(0,1,0));
                glm::quat qz = glm::angleAxis(glm::radians(rot.z), glm::vec3(0,0,1));
                transform.rotation = glm::normalize(qy * qx * qz);
                uint32_t texIndex = instance["texIndex"];
                if (texIndex == UINT32_MAX) texIndex = (baseMaterial + mesh.materialIndex) * cm::Texture::UNKNOWN;
                else texIndex *= cm::Texture::UNKNOWN;
                VkASInstance asInstance{
                    .transform = toVkTransform(transform.matrix()),
                    .instanceCustomIndex = static_cast<uint32_t>(mInstances.size()),
                    .mask = 0xFF,
                    .instanceShaderBindingTableRecordOffset = 0,
                    .flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR,
                    .accelerationStructureReference = blasAddress
                };
                mInstances.push_back(asInstance);
                InstanceInfo instanceInfo {
                    .name = instance["name"],
                    .meshName = mesh.name,
                    .transform = transform,
                    .meshID = static_cast<uint32_t>(mBLASEntries.size() - 1),
                    .textureID = texIndex,
                    .indexCount = static_cast<uint32_t>(indices.size())
                };
                mInstanceInfos.push_back(instanceInfo);
            }
        }
        endCommandBuffer(cmdData, mContext.queue(QueueFamilyType::GRAPHICS));
        mMaterials.insert(mMaterials.end(), loader->materials().begin(), loader->materials().end());
    }

    void PathTracerApp::loadScene() {
        auto directLight = mScene["directLight"];
        mDirectLight.dir = glm::vec4(toVec3(directLight["direction"]), 1);
        mDirectLight.intensity = directLight["intensity"];
        std::vector<std::string> textures = mScene["textureImports"];
        auto materials = mScene["materials"];
        mMaterials.resize(textures.size() + materials.size());
        for (int textureIndex = 0; textureIndex < textures.size(); ++textureIndex) {
            mMaterials[textureIndex].mTextures[cm::Texture::DIFFUSE] =
                cm::AbsLoader::loadTexture(ASSETS_PATH + textures[textureIndex], cm::Texture::DIFFUSE);
            for (int texType = 1; texType < cm::Texture::UNKNOWN; ++texType) {
                mMaterials[textureIndex].mTextures[texType] =
                    cm::AbsLoader::emptyTexture(static_cast<cm::Texture::Type>(texType));
            }
        }
        int baseMaterial = static_cast<int>(textures.size());
        for (int materialIndex = baseMaterial; materialIndex < baseMaterial + materials.size(); ++materialIndex) {
            mMaterials[materialIndex].mTextures[cm::Texture::DIFFUSE] =
                cm::AbsLoader::colorTexture(toVec3(materials[materialIndex]["color"]), cm::Texture::DIFFUSE);
            for (int texType = 1; texType < cm::Texture::UNKNOWN; ++texType) {
                mMaterials[materialIndex].mTextures[texType] =
                    cm::AbsLoader::emptyTexture(static_cast<cm::Texture::Type>(texType));
            }
        }

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

    void PathTracerApp::createRayTracerPass() {
        const RayTracerPassCreateInfo createInfo {
            .context = &mContext,
            .blasInfos = &mBLASInfos,
            .tlas = &mTLAS,
            .instanceInfos = &mInstanceInfos,
            .textures = &mTextures,
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

    void PathTracerApp::createImGui() {
        const ImGuiCreateInfo createInfo {
            .context = &mContext,
            .imageCount = static_cast<uint32_t>(mSwapchainImages.size()),
            .format = mSwapchain.format(),
            .alpha = 0.4f,
            .scale = 1.0f
        };
        mImGui = VkImGui(createInfo);
        VkImGui::loadConfigFile(PROJECT_PATH"imgui.ini");
    }

    void PathTracerApp::recordTracer() {
        VkCommandBuffer commandBuffer = mTracerCommandBuffers[mCurrentFrame];
        vkResetCommandBuffer(commandBuffer, 0);
        beginCommandBuffer(commandBuffer);
        const RayTracerPassRecordInfo recordInfo {
            .commandBuffer = commandBuffer,
            .constants = {
                .frameCount = mFrameCount,
                .spp = static_cast<uint32_t>(mSPP),
                .minDepth = static_cast<uint32_t>(mMinDepth),
                .maxDepth = static_cast<uint32_t>(mMaxDepth),
                .displayMode = static_cast<uint32_t>(mDisplayMode)
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
            const InstanceInfo& instanceInfo = mInstanceInfos[mSelectedInstanceId - 1];
            BLASEntry& blasEntry = mBLASEntries[instanceInfo.meshID];
            recordInfo = {
                .commandBuffer = commandBuffer,
                .vertexBuffer = &blasEntry.vertexBuffer,
                .indexBuffer = &blasEntry.indexBuffer,
                .indexCount = instanceInfo.indexCount,
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
            .currentFrame = mCurrentFrame
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

        if (mRenderImGui) {
            swapchainTransitInfo.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
            swapchainTransitInfo.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            swapchainTransitInfo.srcStage = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        }
        Image::inverseTransit({presentTransitInfo, swapchainTransitInfo});

        if (mRenderImGui) {
            const ImGuiRenderInfo renderInfo {
                .commandBuffer = commandBuffer,
                .imageView = mSwapchainImageViews[imageIndex].get(),
                .extent = mSwapchain.extent()
            };
            mImGui.render(renderInfo);
            swapchainTransitInfo.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
            swapchainTransitInfo.dstAccessMask = 0;
            swapchainTransitInfo.srcStage = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
            swapchainTransitInfo.dstStage = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
            swapchainTransitInfo.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            swapchainTransitInfo.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
            Image::transit(swapchainTransitInfo);
        }
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

    void PathTracerApp::updateSelectedInstance() {
        uint32_t* data = nullptr;
        vmaMapMemory(mContext.allocator(), mReadbackBuffer.allocation(), (void**)&data);
        mSelectedInstanceId = *data;
        vmaUnmapMemory(mContext.allocator(), mReadbackBuffer.allocation());
    }

    void PathTracerApp::update() {
        const RayTracerPassUpdateInfo tracerUpdateInfo {
            .camera = mCamera,
            .directLight = mDirectLight,
            .currentFrame = mCurrentFrame
        };
        mRayTracerPass.update(tracerUpdateInfo);

        if (mSelectedInstanceId == 0) return;
        const RasterizerPassUpdateInfo rasterUpdateInfo {
            .camera = mCamera,
            .instanceInfo = &mInstanceInfos[mSelectedInstanceId - 1],
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
        ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(500, 200), ImGuiCond_FirstUseEver);
        mImGui.beginFrame();
        if (!mRenderImGui) {
            mImGui.endFrame();
            return;
        }

        if (ImGui::Begin("Overview", nullptr, ImGuiWindowFlags_MenuBar)) {
            if (ImGui::BeginMenuBar()) {
                if (ImGui::BeginMenu("File")) {
                    if (ImGui::MenuItem("Save Panel Configuration")) {
                        VkImGui::saveConfigFile(PROJECT_PATH"imgui.ini");
                    }
                    ImGui::EndMenu();
                }
                ImGui::EndMenuBar();
            }

            if (VkImGui::beginGroup(ICON_FA_GAUGE " Status")) {
                std::string fps = std::format("{:.1f}", ImGui::GetIO().Framerate);
                std::string renderTime = std::format("{:.1f} ms", ImGui::GetIO().DeltaTime * 1000.0f);
                std::string accumulation = std::format("{:1}", (mFrameCount + 1) * mSPP);

                if (VkImGui::beginCompactTable("##monitor_status", 2.0f)) {
                    VkImGui::row("FPS"         , fps.c_str());
                    VkImGui::row("Render Time" , renderTime.c_str());
                    VkImGui::row("SPP"         , "1");
                    VkImGui::row("Accumulation", accumulation.c_str());
                    VkImGui::endCompactTable();
                }
                VkImGui::endGroup();
            }

            if (VkImGui::beginGroup(ICON_FA_MICROCHIP " System")) {
                auto properties = mContext.physicalDeviceProperties();
                auto [width, height] = mSwapchain.extent();
                std::string viewport = std::format("{:1}x{:2}", width, height);
                if (VkImGui::beginCompactTable("##monitor_system", 2.0f)) {
                    VkImGui::row("GPU"     , properties.deviceName);
                    VkImGui::row("Viewport", viewport.c_str());
                    VkImGui::endCompactTable();
                }
                VkImGui::endGroup();
            }

            if (VkImGui::beginGroup(ICON_FA_CUBES " Scene")) {
                VkImGui::endGroup();
            }

        }
        ImGui::End();

        if (ImGui::Begin("Settings")) {
            auto cameraFunc = [this] {
            glm::vec3 position = mCamera->position();
            if (ImGui::DragFloat3("Position", &position.x, 0.05f, -FLT_MAX, FLT_MAX)) {
                mCamera->setPosition(position);
                updateImage();
            }
            float fov = mCamera->FOV();
            if (ImGui::SliderFloat("FOV", &fov, 10, 140, "%.2f deg")) {
                mCamera->zoom(mCamera->FOV() - fov);
                updateImage();
            }
            const bool isFlyCamera = mCamera->type() == cs::CameraType::FLY;
            if (VkImGui::selectableButton("Fly", isFlyCamera)) {
                setCamera(scene::CameraType::FLY);
            }
            ImGui::SameLine(0.0f, 5.0f);
            if (VkImGui::selectableButton("Orbital", !isFlyCamera)) {
                setCamera(scene::CameraType::ORBITAL);
                updateImage();
            }
            ImGui::SameLine();
            ImGui::Text("Type");
            };
            auto renderFunc = [this] {
                ImGui::Indent(4.0f);
                if (ImGui::CollapsingHeader("Direct Light", ImGuiTreeNodeFlags_DefaultOpen)) {
                    if (ImGui::DragFloat3("Direction", &mDirectLight.dir.x, 0.005f, -1.0f, 1.0f)) {
                        updateImage();
                    }
                    if (ImGui::DragFloat("Intensity", &mDirectLight.intensity, 0.05f, 0.0f, 10.0f)) {
                        updateImage();
                    }
                }
                if (ImGui::CollapsingHeader("Performance", ImGuiTreeNodeFlags_DefaultOpen)) {
                    if (ImGui::DragInt("SPP", &mSPP, 0.05f, 1, INT_MAX)) {
                        updateImage();
                    }
                    if (ImGui::DragInt("Min Bounces", &mMinDepth, 0.05f, 0, mMaxDepth)) {
                        updateImage();
                    }
                    if (ImGui::DragInt("Max Bounces", &mMaxDepth, 0.05f, 1, INT_MAX)) {
                        updateImage();
                    }
                }
                if (ImGui::CollapsingHeader("Debug", ImGuiTreeNodeFlags_DefaultOpen)) {
                    // const char* modes[] = {"Albedo", "Depth", "Normal", "TracedAlbedo", "Rendered"};
                    // if (ImGui::Combo("Display mode", &mDisplayMode, modes, IM_ARRAYSIZE(modes))) {
                    //     updateImage();
                    // }
                }
                ImGui::Unindent(4.0f);
            };

            auto objectFunc = [this] {
                if (mSelectedInstanceId == 0) {
                    ImGui::Text("Click a mesh in the viewport");
                    return;
                }
                InstanceInfo& instance = mInstanceInfos[mSelectedInstanceId - 1];
                if (VkImGui::beginGroup(ICON_FA_CIRCLE_INFO " Object")) {
                    if (VkImGui::beginCompactTable("##object_status", 6.0f)) {
                        VkImGui::row("Name"         , instance.name.c_str());
                        VkImGui::row("Mesh Name"    , instance.meshName.c_str());
                        VkImGui::row("Texture index", std::to_string(instance.textureID).c_str());
                        VkImGui::endCompactTable();
                    }
                    VkImGui::endGroup();
                }

                if (VkImGui::beginGroup(ICON_FA_ARROWS_UP_DOWN_LEFT_RIGHT " Transform")) {
                    bool changed = false;
                    Transform& transform = instance.transform;
                    if (ImGui::DragFloat3("Position", &transform.position[0], 0.1f))
                        changed = true;

                    static glm::vec3 uiRotation{FLT_MAX};
                    if (uiRotation[0] == FLT_MAX)
                        uiRotation = glm::degrees(glm::eulerAngles(instance.transform.rotation));
                    glm::vec3 prevRotation = uiRotation;
                    if (ImGui::DragFloat3("Rotation", &uiRotation[0], 0.5f)) {
                        uiRotation = clampRotation(uiRotation);
                        glm::vec3 delta = uiRotation - prevRotation;
                        glm::quat qx = glm::angleAxis(glm::radians(delta.x), glm::vec3(1,0,0));
                        glm::quat qy = glm::angleAxis(glm::radians(delta.y), glm::vec3(0,1,0));
                        glm::quat qz = glm::angleAxis(glm::radians(delta.z), glm::vec3(0,0,1));
                        glm::quat deltaRot = qz * qy * qx;
                        transform.rotation = glm::normalize(deltaRot * transform.rotation);
                        changed = true;
                    }

                    if (ImGui::DragFloat3("Scale", &transform.scale[0], 0.05f))
                        changed = true;

                    if (changed) {
                        updateInstanceModel();
                        updateImage();
                    }
                    VkImGui::endGroup();
                }
            };

            static TabPanel panel = {
                {ICON_FA_CUBE   " Object", 0, objectFunc},
                {ICON_FA_IMAGES " Render", 1, renderFunc},
                {ICON_FA_VIDEO  " Camera", 2, cameraFunc},
            };
            static uint32_t activeSettingsTabIndex = 1;
            VkImGui::tabPanel(panel, activeSettingsTabIndex);
        }
        ImGui::End();

        mImGui.endFrame();
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

    void PathTracerApp::updateInstanceModel() {
        const InstanceInfo& instanceInfo = mInstanceInfos[mSelectedInstanceId - 1];
        VkASInstance& asInstance = mInstances[mSelectedInstanceId - 1];
        asInstance.transform = toVkTransform(instanceInfo.transform.matrix());
        const CopyDataToGPUBufferInfo copyInfo {
            .data = mInstances.data(),
            .srcOffset = sizeof(VkASInstance) * (mSelectedInstanceId - 1),
            .dstOffset = sizeof(VkASInstance) * (mSelectedInstanceId - 1),
            .size = sizeof(VkASInstance),
            .allocator = mContext.allocator(),
            .buffer = mInstanceBuffer.get(),
            .device = mContext.device(),
            .queueFamilyIndex = mContext.familyIndex(QueueFamilyType::GRAPHICS).value(),
            .queue = mContext.queue(QueueFamilyType::GRAPHICS)
        };
        Buffer::copy(copyInfo);
        auto [commandBuffer, cmdData] = beginCommandBuffer(&mContext, QueueFamilyType::GRAPHICS);
        const TLASUpdateInfo updateInfo {
            .commandBuffer = commandBuffer,
            .instanceCount = static_cast<uint32_t>(mInstances.size())
        };
        mTLAS.update(updateInfo);
        endCommandBuffer(cmdData, mContext.queue(QueueFamilyType::GRAPHICS));
    }
}
