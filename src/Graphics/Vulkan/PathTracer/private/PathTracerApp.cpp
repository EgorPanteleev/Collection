//
// Created by igor on 4/22/26.
//

#include "PathTracerApp.hpp"
#include "CoreUtils.hpp"
#include "Timer.hpp"
#include "Loader.hpp"
#include "BinnedSAHBuilder.hpp"
#include "CallBacks.hpp"
#include "IconsFontAwesome6.h"
#include <glm/gtx/string_cast.hpp>
#include "TLAS.hpp"

#include <fstream>
#include <filesystem>

namespace fs = std::filesystem;

#define COLOR_FORMAT VK_FORMAT_R8G8B8A8_UNORM
#define NORMAL_FORMAT VK_FORMAT_R8G8B8A8_SNORM
#define INSTANCE_ID_FORMAT VK_FORMAT_R32_UINT

static glm::vec3 toVec3(const nlohmann::json& json) {
    return {
        json[0].get<float>(),
        json[1].get<float>(),
        json[2].get<float>()
    };
}

namespace crv::graphics::vulkan {
    PathTracerApp::PathTracerApp(const PathTracerAppCreateInfo& info) {
        std::ifstream file(info.scenePath);
        file >> mScene;
        createCamera();
        mDirectLight = info.directLight;
        createContext();
        setCallBacks(this);
        createCommandPool();
        createCommandBuffers();
        createSwapChain();
        createSwapChainImages();
        createSyncObjects();
        createImages();

        loadScene();
        createTextures();

        createGBuffers();
        createRasterizer();
        createPathTracer();
        createOutliner();
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

    void PathTracerApp::updateImage() {
        mFrameCount = 0;
    }

    void PathTracerApp::pixelClicked(uint32_t x, uint32_t y) {
        if (!mRenderImGui) return;
        auto [width, height] = mSwapchain.extent();
        if (x > width or y > height) return;
        mPixel = {x, y};
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

    void PathTracerApp::createContext() {
        auto window = mScene["window"];
        const WindowCreateInfo windowCreateInfo {
            .width = window["width"],
            .height = window["height"],
            .name = window["name"]
        };
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
        mTracerCommandPool = CommandPool(createInfo);
        createInfo.queueFamilyIndex = mContext.familyIndex(QueueFamilyType::GRAPHICS).value();
        mRasterCommandPool = CommandPool(createInfo);
        mOutlinerCommandPool = CommandPool(createInfo);
    }

    void PathTracerApp::createCommandBuffers() {
        CommandBuffersCreateInfo createInfo {
            .device = mContext.device(),
            .commandPool = mTracerCommandPool.get(),
            .bufferCount = mFramesInFlight
        };
        mTracerCommandBuffers = CommandBuffers(createInfo);
        createInfo.commandPool = mRasterCommandPool.get();
        mRasterCommandBuffers = CommandBuffers(createInfo);
        createInfo.commandPool = mOutlinerCommandPool.get();
        mOutlinerCommandBuffers = CommandBuffers(createInfo);
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
        mOutlinerFinishedSemaphores.resize(mFramesInFlight);
        mFences.resize(mFramesInFlight);
        for (uint32_t i = 0; i < mFramesInFlight; ++i) {
            mImageAvailableSemaphores[i] = Semaphore(semaphoreCreateInfo);
            mRasterFinishedSemaphores[i] = Semaphore(semaphoreCreateInfo);
            mTracerFinishedSemaphores[i] = Semaphore(semaphoreCreateInfo);
            mOutlinerFinishedSemaphores[i] = Semaphore(semaphoreCreateInfo);
            mFences[i] = Fence(fenceCreateInfo);
        }
    }

    void PathTracerApp::createImages() {
        ImageCreateInfo imageCreateInfo {
            .device = mContext.device(),
            .allocator = mContext.allocator(),
            .flags = 0,
            .format = VK_FORMAT_R8G8B8A8_UNORM,
            .extent = {mSwapchain.extent().width, mSwapchain.extent().height, 1},
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .tiling = VK_IMAGE_TILING_OPTIMAL,
            .imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
            .memoryUsage = VMA_MEMORY_USAGE_AUTO
        };
        mPresentImage  = Image(imageCreateInfo);
        imageCreateInfo.imageUsage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        mTracerImage   = Image(imageCreateInfo);
        ImageViewCreateInfo imageViewCreateInfo {
            .device = mContext.device(),
            .image = mTracerImage.get(),
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format = VK_FORMAT_R8G8B8A8_UNORM,
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .mipLevels = 1,
            .baseMipLevel = 0,
            .baseArrayLayer = 0,
            .layerCount = 1
        };
        mTracerView   = ImageView(imageViewCreateInfo);
        imageViewCreateInfo.image = mPresentImage.get();
        mPresentView  = ImageView(imageViewCreateInfo);

        auto [commandPool, commandBuffers] = beginCommandBuffer(mContext.device(), mContext.familyIndex(QueueFamilyType::GRAPHICS).value());
        VkCommandBuffer commandBuffer = (*commandBuffers)[0];
        const ImageTransitInfo presentTransitInfo {
            .commandBuffer = commandBuffer,
            .image = mPresentImage.get(),
            .srcAccessMask = 0,
            .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            .dstStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
        };
        Image::transit(presentTransitInfo);
        endCommandBuffer(commandPool, commandBuffers, mContext.queue(QueueFamilyType::GRAPHICS));

        #ifndef NDEBUG
            DEBUG << "Tracer image: " << mTracerImage.get();
            DEBUG << "Present image: " << mPresentImage.get();
        #endif
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

    static BLAS buildBLAS(std::span<Tri> tris) {
        BLASBuilder builder{tris};
        BLAS blas = builder.build();
        return blas;
    }

    static TLAS buildTLAS(std::span<MeshPrimitive> prims) {
        TLASBuilder builder{prims};
        TLAS tlas = builder.build();
        return tlas;
    }

    void PathTracerApp::loadScene() {
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

        for (const auto& instance: mRasterInstances) {
            AlignedBBox bbox = mNodes[instance.baseNode].bbox;
            mMeshPrimitives.emplace_back(instance.transform.matrix(), bbox.min, bbox.max);
        }

        TLAS tlas = buildTLAS(std::span(mMeshPrimitives));
        for (int i = 0; i < mRasterInstances.size(); ++i) {
            const uint32_t idx = tlas.primIds()[i];
            mTracerInstances.push_back(mRasterInstances[idx]);
        }
        for (const auto& node: tlas.nodes()) {
            AlignedNode alignedNode{};
            alignedNode.bbox = AlignedBBox(Vec4(node.bbox().min, 1), Vec4(node.bbox().max, 1));
            alignedNode.index = node.index().value();
            mTLASNodes.push_back(alignedNode);
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
        uint32_t instanceCount = instances.size();
        uint32_t baseVertex = mVertices.size();
        uint32_t baseIndex = mIndices.size();
        uint32_t baseMaterial = mMaterials.size();
        mMeshesData.reserve(mMeshesData.capacity() + loader->meshes().size());
        for (size_t meshIndex = 0; meshIndex < loader->meshes().size(); ++meshIndex) {
            const auto& mesh = loader->meshes()[meshIndex];
            MeshData meshData {
                .baseVertex = baseVertex + mesh.baseVertex,
                .baseIndex = baseIndex + mesh.baseIndex,
                .indexCount = mesh.numIndices,
                .baseInstance = static_cast<uint32_t>(mRasterInstances.size()),
                .instanceCount = instanceCount
            };
            mMeshesData.push_back(meshData);
            mVertices.reserve(mVertices.capacity() + mesh.numVertices);
            for (size_t i = 0; i < mesh.numVertices; ++i) {
                const cm::Vertex& modelVertex = loader->vertices()[mesh.baseVertex + i];
                Vertex vertex {
                    .pos = modelVertex.pos,
                    .texCoord = modelVertex.texCoord0,
                    .normal = modelVertex.normal,
                    .tangent = modelVertex.tangent,
                };
                mVertices.push_back(vertex);
            }

            std::vector<Tri> triangles;
            std::vector<uint32_t> indices;
            for (size_t i = 0; i < mesh.numIndices; i += 3) {
                const size_t idx = mesh.baseIndex + i;
                triangles.emplace_back(loader->vertices()[mesh.baseVertex + loader->indices()[idx + 0]].pos,
                                       loader->vertices()[mesh.baseVertex + loader->indices()[idx + 1]].pos,
                                       loader->vertices()[mesh.baseVertex + loader->indices()[idx + 2]].pos);
            }
            BVH blas = buildBLAS(std::span(triangles));

            uint32_t baseTriangle = mTriangles.size();
            uint32_t baseNode = mNodes.size();
            for (int i = 0; i < triangles.size(); ++i) {
                const uint32_t triIdx = blas.primIds()[i];
                const uint32_t idx = mesh.baseIndex + triIdx * 3;
                Tri& tri = triangles[triIdx];
                mTriangles.emplace_back(Vec4(tri.p0, 1), Vec4(tri.e1, 1),
                                        Vec4(tri.e2, 1), Vec4(tri.N, 1));
                cm::Vertex v0 = loader->vertices()[mesh.baseVertex + loader->indices()[idx + 0]];
                cm::Vertex v1 = loader->vertices()[mesh.baseVertex + loader->indices()[idx + 1]];
                cm::Vertex v2 = loader->vertices()[mesh.baseVertex + loader->indices()[idx + 2]];
                mTriangleExtras.emplace_back(v0.texCoord0, v1.texCoord0, v2.texCoord0);
            }
            for (const auto& node: blas.nodes()) {
                AlignedNode alignedNode{};
                alignedNode.bbox = AlignedBBox(Vec4(node.bbox().min, 1), Vec4(node.bbox().max, 1));
                alignedNode.index = node.index().value();
                mNodes.push_back(alignedNode);
            }

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
                MeshInstance meshInstance = {
                    .name = instance["name"],
                    .meshName = mesh.name,
                    .transform = transform,
                    .baseNode = baseNode,
                    .baseTri = baseTriangle,
                    .texIndex = texIndex
                };
                mRasterInstances.push_back(meshInstance);
            }
        }
        mIndices.insert(mIndices.end(), loader->indices().begin(), loader->indices().end());
        mMaterials.insert(mMaterials.end(), loader->materials().begin(), loader->materials().end());
    }

    void PathTracerApp::createPathTracer() {
        const PathTracerCreateInfo pathTracerCreateInfo {
            .context = &mContext,
            .triangles = mTriangles,
            .triangleExtras = mTriangleExtras,
            .nodes = mNodes,
            .TLASNodes = mTLASNodes,
            .textures = &mTextures,
            .instances = mTracerInstances,
            .outImage = mTracerImage.get(),
            .outImageView = mTracerView.get(),
            .gBuffers = &mGBuffers,
            .framesInFlight = mFramesInFlight
        };
        mPathTracer = PathTracer(pathTracerCreateInfo);
    }

    void PathTracerApp::createOutliner() {
        const OutlinerCreateInfo outlinerCreateInfo {
            .context = &mContext,
            .framesInFlight = mFramesInFlight
        };
        mOutliner = Outliner(outlinerCreateInfo);
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
            .currentFrame = mCurrentFrame
        };
        mPathTracer.update(pathTracerUpdateInfo);

        const OutlinerUpdateInfo outlinerUpdateInfo {
            .tracerImageView = mTracerView.get(),
            .instanceIdImageView = currentGBuffer().selectedInstanceView.get(),
            .tracerSampler = currentGBuffer().sampler.get(),
            .instanceIdSampler = currentGBuffer().intSampler.get()
        };
        mOutliner.update(outlinerUpdateInfo);
    }

    void PathTracerApp::recordRaster() {
        VkCommandBuffer commandBuffer = mRasterCommandBuffers[mCurrentFrame];
        vkResetCommandBuffer(commandBuffer, 0);
        beginCommandBuffer(commandBuffer);
        const RasterizerRecordInfo rasterizerRecordInfo {
            .commandBuffer = commandBuffer,
            .gBuffer = &mGBuffers[mCurrentFrame],
            .extent = mSwapchain.extent(),
            .currentFrame = mCurrentFrame,
            .clickPos = mPixel
        };
        mPixel = {UINT32_MAX, UINT32_MAX};
        const ImageTransitInfo selectedInstanceTransitInfo {
            .commandBuffer = commandBuffer,
            .image = currentGBuffer().selectedInstanceImage.get(),
            .srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .srcStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            .dstStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
        };

        Image::inverseTransit(selectedInstanceTransitInfo);
        mRasterizer.record(rasterizerRecordInfo);
        Image::transit(selectedInstanceTransitInfo);

        GBuffer& gBuffer = currentGBuffer();
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

    void PathTracerApp::recordTracer() {
        VkCommandBuffer commandBuffer = mTracerCommandBuffers[mCurrentFrame];
        vkResetCommandBuffer(commandBuffer, 0);
        beginCommandBuffer(commandBuffer);

        const PathTracerRecordInfo pathTracerRecordInfo {
            .commandBuffer = commandBuffer,
            .extent = mSwapchain.extent(),
            .currentFrame = mCurrentFrame,
            .constants = {
                .frameCount = mFrameCount,
                .spp = static_cast<uint32_t>(mSPP),
                .minDepth = static_cast<uint32_t>(mMinDepth),
                .maxDepth = static_cast<uint32_t>(mMaxDepth),
                .displayMode = static_cast<uint32_t>(mDisplayMode)
            }
        };
        mPathTracer.record(pathTracerRecordInfo);

        GBuffer& gBuffer = currentGBuffer();
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
        endCommandBuffer(commandBuffer);
    }

    void PathTracerApp::recordOutliner(const uint32_t imageIndex) {
        VkCommandBuffer commandBuffer = mOutlinerCommandBuffers[mCurrentFrame];
        vkResetCommandBuffer(commandBuffer, 0);
        beginCommandBuffer(commandBuffer);
        const OutlinerRecordInfo recordInfo {
            .commandBuffer = commandBuffer,
            .outImageView = mPresentView.get(),
            .extent = mSwapchain.extent(),
            .currentFrame = mCurrentFrame
        };
        mOutliner.record(recordInfo);
        recordPresent(imageIndex, commandBuffer);
        endCommandBuffer(commandBuffer);
    }

    void PathTracerApp::recordPresent(uint32_t imageIndex, VkCommandBuffer commandBuffer) {
        const ImageBarrierInfo dataBarrierInfo {
            .image = mPresentImage.get(),
            .srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
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
            .srcStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
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

        VkImageMemoryBarrier inverseDataBarrier = Image::inverseBarrier(dataBarrierInfo);
        pipelineBarrierInfo1.barriers = {inverseDataBarrier};
        Image::inversePipelineBarrier(pipelineBarrierInfo1);

        VkImageMemoryBarrier inversePresentBarrier = Image::inverseBarrier(presentBarrierInfo);

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

    void PathTracerApp::record(uint32_t imageIndex) {
        recordRaster();
        recordTracer();
        recordOutliner(imageIndex);
        vkResetFences(mContext.device(), 1, &mFences[mCurrentFrame].get());
    }

    void PathTracerApp::submit(const uint32_t imageIndex) {
        auto& rasterCommandBuffer = mRasterCommandBuffers[mCurrentFrame];
        VkPipelineStageFlags rasterWaitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
        const VkSubmitInfo rasterSubmitInfo{
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &mImageAvailableSemaphores[mCurrentFrame].get(),
            .pWaitDstStageMask = rasterWaitStages,
            .commandBufferCount = 1,
            .pCommandBuffers = &rasterCommandBuffer,
            .signalSemaphoreCount = 1,
            .pSignalSemaphores = &mRasterFinishedSemaphores[mCurrentFrame].get()
        };

        VkQueue graphicsQueue = mContext.queue(QueueFamilyType::GRAPHICS);
        if (vkQueueSubmit(graphicsQueue, 1, &rasterSubmitInfo, VK_NULL_HANDLE) != VK_SUCCESS) {
            throw std::runtime_error("Failed to submit draw command buffer!");
        }

        auto& tracerCommandBuffer = mTracerCommandBuffers[mCurrentFrame];
        VkPipelineStageFlags tracerWaitStages[] = {VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT};
        const VkSubmitInfo tracerSubmitInfo{
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &mRasterFinishedSemaphores[mCurrentFrame].get(),
            .pWaitDstStageMask = tracerWaitStages,
            .commandBufferCount = 1,
            .pCommandBuffers = &tracerCommandBuffer,
            .signalSemaphoreCount = 1,
            .pSignalSemaphores = &mTracerFinishedSemaphores[mCurrentFrame].get()
        };
        VkQueue computeQueue = mContext.queue(QueueFamilyType::COMPUTE);
        if (vkQueueSubmit(computeQueue, 1, &tracerSubmitInfo, VK_NULL_HANDLE) != VK_SUCCESS) {
            throw std::runtime_error("Failed to submit draw command buffer!");
        }

        auto& outlinerCommandBuffer = mOutlinerCommandBuffers[mCurrentFrame];
        VkPipelineStageFlags outlinerWaitStages[] = {VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT};
        const VkSubmitInfo outlinerSubmitInfo{
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &mTracerFinishedSemaphores[mCurrentFrame].get(),
            .pWaitDstStageMask = outlinerWaitStages,
            .commandBufferCount = 1,
            .pCommandBuffers = &outlinerCommandBuffer,
            .signalSemaphoreCount = 1,
            .pSignalSemaphores = &mOutlinerFinishedSemaphores[mCurrentFrame].get()
        };
        if (vkQueueSubmit(computeQueue, 1, &outlinerSubmitInfo, mFences[mCurrentFrame].get()) != VK_SUCCESS) {
            throw std::runtime_error("Failed to submit draw command buffer!");
        }
        const VkPresentInfoKHR presentInfo {
            .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &mOutlinerFinishedSemaphores[mCurrentFrame].get(),
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
        mRasterizer.updateSelectedInstance();
        acquireNextImage(imageIndex);
        drawControlPanel();
        update();
        record(imageIndex);
        submit(imageIndex);
    }

    static float clampAngle(float deg) {
        deg = std::fmod(deg + 180.0f, 360.0f);
        if (deg < 0.0f) deg += 360.0f;
        return deg - 180.0f;
    }

    static glm::vec3 clampRotation(const glm::vec3& e) {
        return {clampAngle(e.x), clampAngle(e.y), clampAngle(e.z)};
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
                    const char* modes[] = {"Albedo", "Depth", "Normal", "TracedAlbedo", "Rendered"};
                    if (ImGui::Combo("Display mode", &mDisplayMode, modes, IM_ARRAYSIZE(modes))) {
                        updateImage();
                    }
                }
                ImGui::Unindent(4.0f);
            };

            auto objectFunc = [this] {
                uint32_t selectedInstanceIdx = mRasterizer.selectedInstanceIdx();
                if (selectedInstanceIdx == UINT32_MAX) {
                    ImGui::Text("Click a mesh in the viewport");
                    return;
                }

                MeshInstance& instance = mRasterInstances[selectedInstanceIdx];
                if (VkImGui::beginGroup(ICON_FA_CIRCLE_INFO " Object")) {
                    if (VkImGui::beginCompactTable("##object_status", 6.0f)) {
                        VkImGui::row("Name"         , instance.name.c_str());
                        VkImGui::row("Mesh Name"    , instance.meshName.c_str());
                        VkImGui::row("Texture index", std::to_string(instance.texIndex).c_str());
                        VkImGui::endCompactTable();
                    }
                    VkImGui::endGroup();
                }

                if (VkImGui::beginGroup(ICON_FA_ARROWS_UP_DOWN_LEFT_RIGHT " Transform")) {
                    bool changed = false;
                    Transform transform = instance.transform;
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
                        updateInstanceModel(selectedInstanceIdx, transform);
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
        const ImageCreateInfo selectedInstanceImageCreateInfo {
            .device = mContext.device(),
            .allocator = mContext.allocator(),
            .format = INSTANCE_ID_FORMAT,
            .extent = extent,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .tiling = VK_IMAGE_TILING_OPTIMAL,
            .imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            .memoryUsage = VMA_MEMORY_USAGE_AUTO
        };
        ImageViewCreateInfo selectedInstanceViewCreateInfo {
            .device = mContext.device(),
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format = INSTANCE_ID_FORMAT,
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

        const SamplerCreateInfo intSamplerCreateInfo {
            .device = mContext.device(),
            .physicalDevice = mContext.physicalDevice(),
            .addressMode = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            .compareEnable = VK_FALSE,
            .compareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
            .mipLevels = 1,
            .borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK,
            .magFilter = VK_FILTER_NEAREST,
            .minFilter = VK_FILTER_NEAREST,
            .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST
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
        constexpr ImageBarrierInfo shaderBarrierInfo {
            .srcAccessMask = 0,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
        };
        VkImageMemoryBarrier colorBarrier = Image::barrier(colorBarrierInfo);
        VkImageMemoryBarrier depthBarrier = Image::barrier(depthBarrierInfo);
        VkImageMemoryBarrier shaderBarrier = Image::barrier(shaderBarrierInfo);

        std::vector<VkImageMemoryBarrier> colorBarriers;
        std::vector<VkImageMemoryBarrier> depthBarriers;
        std::vector<VkImageMemoryBarrier> shaderBarriers;
        mGBuffers.resize(mFramesInFlight);
        for (int i = 0; i < mFramesInFlight; ++i) {
            GBuffer& gBuffer = mGBuffers[i];
            std::vector images = {
                &gBuffer.colorImage , &gBuffer.depthImage,
                &gBuffer.normalImage, &gBuffer.selectedInstanceImage
            };
            std::vector views = {
                &gBuffer.colorView , &gBuffer.depthView,
                &gBuffer.normalView, &gBuffer.selectedInstanceView
            };
            std::vector imageInfos = {
                colorImageCreateInfo, depthImageCreateInfo,
                normalImageCreateInfo, selectedInstanceImageCreateInfo
            };
            std::vector viewInfos = {
                colorViewCreateInfo, depthViewCreateInfo,
                normalViewCreateInfo, selectedInstanceViewCreateInfo
            };
            #ifndef NDEBUG
                std::vector<std::string> names = {"Color", "Depth", "Normal", "Selected instance"};
                DEBUG << "Frame in flight: " << i;
            #endif
            for (size_t j = 0; j < images.size(); ++j) {
                *images[j] = Image(imageInfos[j]);
                viewInfos[j].image = images[j]->get();
                *views[j] = ImageView(viewInfos[j]);
                #ifndef NDEBUG
                    DEBUG << names[j] + " image: " << images[j]->get();
                    DEBUG << names[j] + " view: " << views[j]->get();
                #endif
            }
            gBuffer.sampler = Sampler(samplerCreateInfo);
            gBuffer.intSampler = Sampler(intSamplerCreateInfo);

            depthBarrier.image = gBuffer.depthImage.get();
            depthBarriers.push_back(depthBarrier);
            colorBarrier.image = gBuffer.colorImage.get();
            colorBarriers.push_back(colorBarrier);
            colorBarrier.image = gBuffer.normalImage.get();
            colorBarriers.push_back(colorBarrier);
            shaderBarrier.image = gBuffer.selectedInstanceImage.get();
            shaderBarriers.push_back(shaderBarrier);
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
        const ImagePipelineBarrierInfo shaderPipelineBarrierInfo {
            .commandBuffer = commandBuffer,
            .srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            .dstStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            .barriers = shaderBarriers
        };
        Image::pipelineBarrier(colorPipelineBarrierInfo);
        Image::pipelineBarrier(depthPipelineBarrierInfo);
        Image::pipelineBarrier(shaderPipelineBarrierInfo);
        endCommandBuffer(commandPool, commandBuffers, mContext.queue(QueueFamilyType::GRAPHICS));
    }

    void PathTracerApp::createRasterizer() {
        const auto [width, height] = mSwapchain.extent();
        const RasterizerCreateInfo createInfo {
            .context = &mContext,
            .colorFormat = COLOR_FORMAT,
            .normalFormat = NORMAL_FORMAT,
            .instanceIdFormat = INSTANCE_ID_FORMAT,
            .extent = {width, height, 1},
            .framesInFlight = mFramesInFlight,
            .vertices = mVertices,
            .indices = mIndices,
            .meshesData = mMeshesData,
            .instances = mRasterInstances,
            .textures = &mTextures
        };
        mRasterizer = Rasterizer(createInfo);
    }

    void PathTracerApp::updateInstanceModel(const uint32_t instanceIndex, const Transform& transform) {
        MeshInstance& instance = mRasterInstances[instanceIndex];
        instance.transform = transform;
        mRasterizer.updateInstanceBuffer(mRasterInstances);
        glm::mat4 model = transform.matrix();

        AlignedBBox bbox = mNodes[instance.baseNode].bbox;
        mMeshPrimitives[instanceIndex] = MeshPrimitive(model, bbox.min, bbox.max);

        TLAS tlas = buildTLAS(std::span(mMeshPrimitives));
        mTracerInstances.clear();
        for (int i = 0; i < mRasterInstances.size(); ++i) {
            const uint32_t idx = tlas.primIds()[i];
            mTracerInstances.push_back(mRasterInstances[idx]);
        }
        mTLASNodes.clear();
        for (const auto& node: tlas.nodes()) {
            AlignedNode alignedNode{};
            alignedNode.bbox = AlignedBBox(Vec4(node.bbox().min, 1), Vec4(node.bbox().max, 1));
            alignedNode.index = node.index().value();
            mTLASNodes.push_back(alignedNode);
        }
        mPathTracer.updateTLAS(mTLASNodes, mTracerInstances, mGBuffers);
    }
}