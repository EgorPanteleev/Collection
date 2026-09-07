//
// Created by igor on 6/12/26.
//

#include "InputHandlers/AppInputHandler.hpp"
#include "PathTracerApp.hpp"
#include "CoreUtils.hpp"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <filesystem>
#include <limits>

namespace fs = std::filesystem;

namespace crv::graphics::vulkan {
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

    void AppInputHandler::apply(const Command& command, PathTracerApp* app) const {
        switch (command.type) {
            case CommandType::SET_CAMERA_FLY:     setCamera(app, scene::CameraType::FLY);     break;
            case CommandType::SET_CAMERA_ORBITAL: setCamera(app, scene::CameraType::ORBITAL); break;

            case CommandType::PICK_OBJECT:     pick(app);           break;
            case CommandType::CLEAR_SELECTION: clearSelection(app); break;
            case CommandType::SELECT_INSTANCE: {
                const auto& p = std::get<SelectInstancePayload>(command.payload);
                selectInstance(app, p.index, p.additive);
                break;
            }
            case CommandType::REGION_SELECT: {
                const auto& p = std::get<RegionSelectPayload>(command.payload);
                regionSelect(app, p.x0, p.y0, p.x1, p.y1, p.additive);
                break;
            }
            case CommandType::DUPLICATE_INSTANCES:
                duplicateInstances(app, std::get<InstancesPayload>(command.payload).indices);
                break;
            case CommandType::REMOVE_INSTANCES:
                removeInstances(app, std::get<InstancesPayload>(command.payload).indices);
                break;
            case CommandType::ADD_MATERIAL:
                addMaterial(app, std::get<MaterialPayload>(command.payload).instanceIndex);
                break;
            case CommandType::UPLOAD_TEXTURE: {
                const auto& p = std::get<UploadTexturePayload>(command.payload);
                uploadTexture(app, p.path, p.materialIndex, p.textureType);
                break;
            }
            case CommandType::LOAD_SKYBOX:
                loadSkybox(app, std::get<SkyboxPayload>(command.payload).path);
                break;
            case CommandType::REMOVE_SKYBOX: removeSkybox(app); break;
            case CommandType::UPDATE_INSTANCE_TRANSFORM:
                for (const uint32_t index : std::get<InstancesPayload>(command.payload).indices)
                    updateInstanceTransform(app, index);
                break;
            case CommandType::UPDATE_INSTANCE:
                updateInstance(app, std::get<IndexPayload>(command.payload).index);
                break;
            case CommandType::UPDATE_MATERIAL:
                updateMaterial(app, std::get<IndexPayload>(command.payload).index);
                break;

            case CommandType::UPDATE_IMAGE:         updateImage(app);        break;
            case CommandType::TOGGLE_CONTROL_PANEL: toggleControlPanel(app); break;

            case CommandType::QUIT:       app->mContext.window().close(); break;
            case CommandType::SAVE_IMAGE: saveImage(app);                 break;
            case CommandType::SAVE_SCENE: saveScene(app);                 break;
            default: break;
        }
    }

    void AppInputHandler::setCamera(PathTracerApp* app, const cs::CameraType type) const {
        Model& model = app->mModel;
        if (type == scene::CameraType::FLY) {
            model.setCamera(&model.flyCamera());
            model.camera()->setPosition(model.orbitalCamera().position());
            model.camera()->setOrientation(model.orbitalCamera().orientation());
        } else {
            model.setCamera(&model.orbitalCamera());
        }
    }

    void AppInputHandler::pick(PathTracerApp* app) const {
        const glm::dvec2 cursor = app->mInput.cursorPos();
        GLFWwindow* window = app->mContext.window().glfwWindow();
        int winWidth, winHeight, fbWidth, fbHeight;
        glfwGetWindowSize(window, &winWidth, &winHeight);
        glfwGetFramebufferSize(window, &fbWidth, &fbHeight);
        const float scaleX = static_cast<float>(fbWidth)  / static_cast<float>(winWidth);
        const float scaleY = static_cast<float>(fbHeight) / static_cast<float>(winHeight);
        const bool additive = app->mInput.isPressed(Key::LEFT_SHIFT) || app->mInput.isPressed(Key::RIGHT_SHIFT);
        const auto x = static_cast<uint32_t>(cursor.x * scaleX);
        const auto y = static_cast<uint32_t>(cursor.y * scaleY);
        auto [width, height] = app->mSwapchain.extent();
        if (x > width or y > height) return;
        app->mClickedPixel = {x, y};
        app->mAdditiveSelect = additive;
    }

    void AppInputHandler::clearSelection(PathTracerApp* app) const {
        app->mModel.selection().selectedInstances.clear();
        app->mModel.selection().activeInstance = UINT32_MAX;
        app->mModel.selection().pending = false;
    }

    void AppInputHandler:: selectInstance(PathTracerApp* app, const uint32_t index, const bool additive) const {
        applySelection(app, index + 1, additive);
    }

    void AppInputHandler::applySelection(PathTracerApp* app, const uint32_t id, const bool additive) const {
        if (id == 0) {
            if (!additive) clearSelection(app);
            return;
        }
        const uint32_t index = id - 1;
        const auto it = std::find(app->mModel.selection().selectedInstances.begin(), app->mModel.selection().selectedInstances.end(), index);
        if (!additive) {
            app->mModel.selection().selectedInstances = {index};
            app->mModel.selection().activeInstance = index;
        } else if (it != app->mModel.selection().selectedInstances.end()) {
            app->mModel.selection().selectedInstances.erase(it);
            app->mModel.selection().activeInstance = app->mModel.selection().selectedInstances.empty() ? UINT32_MAX : app->mModel.selection().selectedInstances.back();
        } else {
            app->mModel.selection().selectedInstances.push_back(index);
            app->mModel.selection().activeInstance = index;
        }
    }

    void AppInputHandler::regionSelect(PathTracerApp* app, int x0, int y0, int x1, int y1, bool additive) const {
        auto [width, height] = app->mSwapchain.extent();
        const float fw = static_cast<float>(width);
        const float fh = static_cast<float>(height);
        const float nx0 = static_cast<float>(std::min(x0, x1)) / fw * 2.0f - 1.0f;
        const float nx1 = static_cast<float>(std::max(x0, x1)) / fw * 2.0f - 1.0f;
        const float ny0 = static_cast<float>(std::min(y0, y1)) / fh * 2.0f - 1.0f;
        const float ny1 = static_cast<float>(std::max(y0, y1)) / fh * 2.0f - 1.0f;

        const glm::mat4 viewProj = app->mModel.camera()->projectionMatrix() * app->mModel.camera()->viewMatrix();
        const auto& instances = app->mModel.scene().mInstances;
        const auto& blasDatas = app->mResourceManager.blasDatas();

        if (!additive) app->mModel.selection().selectedInstances.clear();
        for (uint32_t i = 0; i < instances.size(); ++i) {
            const InstanceData& instance = instances[i];
            const BLASData& mesh = blasDatas[instance.meshIndex];
            const glm::mat4 mvp = viewProj * instance.transform.matrix();

            glm::vec2 boxMin(std::numeric_limits<float>::max());
            glm::vec2 boxMax(std::numeric_limits<float>::lowest());
            bool anyInFront = false;
            for (int c = 0; c < 8; ++c) {
                const glm::vec4 corner {
                    (c & 1) ? mesh.bbox.max.x : mesh.bbox.min.x,
                    (c & 2) ? mesh.bbox.max.y : mesh.bbox.min.y,
                    (c & 4) ? mesh.bbox.max.z : mesh.bbox.min.z,
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

            if (std::find(app->mModel.selection().selectedInstances.begin(), app->mModel.selection().selectedInstances.end(), i) == app->mModel.selection().selectedInstances.end())
                app->mModel.selection().selectedInstances.push_back(i);
        }
        app->mModel.selection().activeInstance = app->mModel.selection().selectedInstances.empty() ? UINT32_MAX : app->mModel.selection().selectedInstances.back();
    }

    void AppInputHandler::duplicateInstances(PathTracerApp* app, const std::vector<uint32_t>& indices) const {
        vkDeviceWaitIdle(app->mContext.device());
        const std::vector<uint32_t> created = app->mResourceManager.duplicateInstances(indices);
        app->mRayTracerPass.bindInstances();
        app->mModel.selection().selectedInstances = created;
        app->mModel.selection().activeInstance = created.empty() ? UINT32_MAX : created.back();
        app->mModel.selection().pending = false;
        updateImage(app);
    }

    void AppInputHandler::removeInstances(PathTracerApp* app, const std::vector<uint32_t>& indices) const {
        vkDeviceWaitIdle(app->mContext.device());
        app->mResourceManager.removeInstances(indices);
        app->mRayTracerPass.bindInstances();
        clearSelection(app);
        updateImage(app);
    }

    void AppInputHandler::addMaterial(PathTracerApp* app, const uint32_t instanceIndex) const {
        vkDeviceWaitIdle(app->mContext.device());
        auto& instances = app->mModel.scene().mInstances;
        if (instanceIndex >= instances.size()) return;
        Material newMaterial = app->mModel.scene().mMaterials[instances[instanceIndex].materialIndex];
        newMaterial.name += " copy";
        const uint32_t index = app->mResourceManager.addMaterial(newMaterial);
        instances[instanceIndex].materialIndex = index;
        app->mResourceManager.updateInstance(instanceIndex);
        app->mRayTracerPass.bindMaterials();
        updateImage(app);
    }

    void AppInputHandler::uploadTexture(PathTracerApp* app, const std::string& path,
                                        const uint32_t materialIndex, const int textureType) const {
        vkDeviceWaitIdle(app->mContext.device());
        uint32_t index = 0;
        switch (textureType) {
            case 1:  index = app->mResourceManager.addNormalTexture(path, materialIndex); break;
            case 2:  index = app->mResourceManager.addMetalRoughnessTexture(path, materialIndex); break;
            case 3:  index = app->mResourceManager.addClearcoatTexture(path, materialIndex); break;
            case 4:  index = app->mResourceManager.addClearcoatRoughnessTexture(path, materialIndex); break;
            default: index = app->mResourceManager.addBaseColorTexture(path, materialIndex); break;
        }
        app->mRayTracerPass.bindTexture(index);
    }

    void AppInputHandler::loadSkybox(PathTracerApp* app, const std::string& path) const {
        vkDeviceWaitIdle(app->mContext.device());
        const uint32_t index = app->mResourceManager.addSkybox(path);
        app->mRayTracerPass.bindTexture(index);
        updateImage(app);
    }

    void AppInputHandler::removeSkybox(PathTracerApp* app) const {
        app->mResourceManager.removeSkybox();
        updateImage(app);
    }

    void AppInputHandler::updateInstanceTransform(PathTracerApp* app, const uint32_t index) const {
        app->mResourceManager.updateInstanceTransform(index);
    }

    void AppInputHandler::updateInstance(PathTracerApp* app, const uint32_t index) const {
        app->mResourceManager.updateInstance(index);
    }

    void AppInputHandler::updateMaterial(PathTracerApp* app, const uint32_t index) const {
        app->mResourceManager.updateMaterial(index);
    }

    void AppInputHandler::updateImage(PathTracerApp* app) const {
        app->mFrameCount = 0;
    }

    void AppInputHandler::toggleControlPanel(PathTracerApp* app) const {
        app->mRenderImGui = !app->mRenderImGui;
    }

    void AppInputHandler::saveImage(PathTracerApp* app) const {
        vkDeviceWaitIdle(app->mContext.device());
        auto [width, height] = app->mSwapchain.extent();

        ImageCreateInfo saveImageInfo {
            .device = app->mContext.device(),
            .allocator = app->mContext.allocator(),
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
            .allocator = app->mContext.allocator(),
            .size = bufferSize,
            .bufferUsage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .memoryUsage = VMA_MEMORY_USAGE_CPU_ONLY
        };
        Buffer buffer(bufferInfo);

        auto [commandBuffer, cmdData] = beginCommandBuffer(app->mContext.device(),
                                            app->mContext.familyIndex(QueueFamilyType::GRAPHICS).value());

        const ImageTransitInfo2 finalTransitInfo {
            .commandBuffer = commandBuffer,
            .image = app->mFinalImage.get(),
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
            app->mFinalImage.get(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
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
        endCommandBuffer(cmdData, app->mContext.queue(QueueFamilyType::GRAPHICS));

        const fs::path outputDir = fs::path(PROJECT_PATH) / "screenshots";
        fs::create_directories(outputDir);
        const auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        char stamp[32];
        std::strftime(stamp, sizeof(stamp), "%Y%m%d_%H%M%S", std::localtime(&now));
        const std::string path = (outputDir / ("render_" + std::string(stamp) + ".png")).string();

        uint8_t* data = nullptr;
        vmaMapMemory(app->mContext.allocator(), buffer.allocation(), reinterpret_cast<void**>(&data));
        const int ok = stbi_write_png(path.c_str(), static_cast<int>(width), static_cast<int>(height),
                                      4, data, static_cast<int>(width) * 4);
        vmaUnmapMemory(app->mContext.allocator(), buffer.allocation());

        if (ok) INFO << "Saved image: " << path;
        else    ERROR << "Failed to save image: " << path;
    }

    void AppInputHandler::saveScene(PathTracerApp* app) const {
        const json scene = app->mModel.scene().save();
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
}
