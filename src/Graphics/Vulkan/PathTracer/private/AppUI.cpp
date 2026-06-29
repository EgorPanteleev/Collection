//
// Created by igor on 6/12/26.
//

#include "AppUI.hpp"
#include "IconsFontAwesome6.h"
#include <ImGuizmo.h>
#include <algorithm>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/quaternion.hpp>

static float clampAngle(float deg) {
    deg = std::fmod(deg + 180.0f, 360.0f);
    if (deg < 0.0f) deg += 360.0f;
    return deg - 180.0f;
}


static glm::vec3 clampRotation(const glm::vec3& e) {
    return {clampAngle(e.x), clampAngle(e.y), clampAngle(e.z)};
}

namespace crv::graphics::vulkan {
    AppUI::AppUI(const AppUICreateInfo& info):
    mContext(info.context), mSwapchain(info.swapchain), mResourceManager(info.resourceManager) {
        const auto [capabilities, formats, presentModes] =
            Swapchain::getSupport(mContext->physicalDevice(), mContext->surface());
        const ImGuiCreateInfo createInfo {
            .context = mContext,
            .imageCount = mSwapchain->getImageCount(capabilities),
            .format = mSwapchain->format(),
            .alpha = 0.4f,
            .scale = 1.0f
        };
        mImGui = VkImGui(createInfo);
        VkImGui::loadConfigFile(PROJECT_PATH"imgui.ini");
    }

    void AppUI::record(const AppUIRecordInfo& info) {
        const ImGuiRenderInfo renderInfo {
            .commandBuffer = info.commandBuffer,
            .imageView = info.imageView->get(),
            .extent = mSwapchain->extent()
        };
        mImGui.render(renderInfo);
    }

    void AppUI::draw(const AppUIDrawInfo& info) {
        ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(500, 200), ImGuiCond_FirstUseEver);
        mImGui.beginFrame();
        if (info.selectedInstances && !info.selectedInstances->empty()) {
            drawGizmo(info);
        }
        handleMarquee();
        if (!info.drawUI) {
            drawCursorDot();
            mImGui.endFrame();
            return;
        }
        drawOverView(info);
        drawSettings(info);
        mImGui.endFrame();
    }

    void AppUI::handleMarquee() {
        ImGuiIO& io = ImGui::GetIO();
        const bool blocked = io.WantCaptureMouse || ImGuizmo::IsOver() || ImGuizmo::IsUsing();
        if (!mMarqueeActive) {
            if (!blocked && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
                mMarqueeActive = true;
                mMarqueeStart = io.MousePos;
            }
            return;
        }

        const ImVec2 cur = io.MousePos;
        const ImVec2 a(std::min(mMarqueeStart.x, cur.x), std::min(mMarqueeStart.y, cur.y));
        const ImVec2 b(std::max(mMarqueeStart.x, cur.x), std::max(mMarqueeStart.y, cur.y));
        ImDrawList* drawList = ImGui::GetForegroundDrawList();
        drawList->AddRectFilled(a, b, IM_COL32(95, 40, 120, 40));
        drawList->AddRect(a, b, IM_COL32(95, 40, 120, 200));

        if (ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
            mMarqueeActive = false;
            if (b.x - a.x > 3.0f && b.y - a.y > 3.0f && mRegionSelect) {
                const ImVec2 scale = io.DisplayFramebufferScale;
                mRegionSelect(static_cast<int>(a.x * scale.x), static_cast<int>(a.y * scale.y),
                              static_cast<int>(b.x * scale.x), static_cast<int>(b.y * scale.y), io.KeyShift);
            }
        }
    }

    void AppUI::drawCursorDot() {
        if (!ImGui::IsMousePosValid()) return;
        ImGui::SetMouseCursor(ImGuiMouseCursor_None);
        ImGui::GetForegroundDrawList()->AddCircleFilled(ImGui::GetIO().MousePos, 3.0f, IM_COL32(95, 40, 120, 255));
    }

    void AppUI::drawGizmo(const AppUIDrawInfo& info) {
        if (info.activeInstance == UINT32_MAX) return;
        if (!ImGui::GetIO().WantTextInput) {
            if (ImGui::IsKeyPressed(ImGuiKey_T)) mGizmoOp = ImGuizmo::TRANSLATE;
            if (ImGui::IsKeyPressed(ImGuiKey_R)) mGizmoOp = ImGuizmo::ROTATE;
        }

        auto& instances = mResourceManager->instances();
        const Transform& pivot = instances[info.activeInstance].transform;
        glm::mat4 view  = info.camera->viewMatrix();
        glm::mat4 proj  = info.camera->projectionMatrix();
        proj[1][1] *= -1.0f;
        glm::mat4 model = glm::translate(glm::mat4(1.0f), pivot.position) * glm::toMat4(pivot.rotation);

        const ImVec2 displaySize = ImGui::GetIO().DisplaySize;
        const ImGuizmo::MODE mode = mGizmoOp == ImGuizmo::ROTATE ? ImGuizmo::LOCAL : ImGuizmo::WORLD;
        ImGuizmo::SetOrthographic(false);
        ImGuizmo::SetDrawlist(ImGui::GetForegroundDrawList());
        ImGuizmo::SetRect(0.0f, 0.0f, displaySize.x, displaySize.y);

        glm::mat4 delta(1.0f);
        if (ImGuizmo::Manipulate(glm::value_ptr(view), glm::value_ptr(proj),
            mGizmoOp, mode, glm::value_ptr(model), glm::value_ptr(delta))) {
            for (const uint32_t index : *info.selectedInstances) {
                Transform& transform = instances[index].transform;
                const glm::mat4 updated = delta *
                    (glm::translate(glm::mat4(1.0f), transform.position) * glm::toMat4(transform.rotation));
                transform.position = glm::vec3(updated[3]);
                transform.rotation = glm::normalize(glm::quat_cast(glm::mat3(updated)));
                mResourceManager->updateInstanceTransform(index);
            }
            mUpdateImage();
        }
    }

    void AppUI::drawOverView(const AppUIDrawInfo& info) {
        if (ImGui::Begin("Overview", nullptr, ImGuiWindowFlags_MenuBar)) {
            if (ImGui::BeginMenuBar()) {
                if (ImGui::BeginMenu("File")) {
                    if (ImGui::MenuItem(ICON_FA_CAMERA " Save Image") && mSaveImage) {
                        mSaveImage();
                    }
                    if (ImGui::MenuItem(ICON_FA_FLOPPY_DISK " Save Scene") && mSaveScene) {
                        mSaveScene();
                    }
                    ImGui::Separator();
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
                std::string accumulation = std::format("{:1}", (info.frameCount + 1) * mSPP);

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
                auto properties = mContext->physicalDeviceProperties();
                auto [width, height] = mSwapchain->extent();
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
    }

    void AppUI::drawCameraTab(const AppUIDrawInfo &info) {
        glm::vec3 position = info.camera->position();
        if (ImGui::DragFloat3("Position", &position.x, 0.05f, -FLT_MAX, FLT_MAX)) {
            info.camera->setPosition(position);
            mUpdateImage();
        }
        float fov = info.camera->FOV();
        if (ImGui::SliderFloat("FOV", &fov, 10, 140, "%.2f deg")) {
            info.camera->zoom(info.camera->FOV() - fov);
            mUpdateImage();
        }
        const bool isFlyCamera = info.camera->type() == cs::CameraType::FLY;
        if (VkImGui::selectableButton("Fly", isFlyCamera)) {
            mCameraSet(scene::CameraType::FLY);
        }
        ImGui::SameLine(0.0f, 5.0f);
        if (VkImGui::selectableButton("Orbital", !isFlyCamera)) {
            mCameraSet(scene::CameraType::ORBITAL);
            mUpdateImage();
        }
        ImGui::SameLine();
        ImGui::Text("Type");
    }

    void AppUI::drawRenderTab(const AppUIDrawInfo& info) {
        ImGui::Indent(4.0f);
        if (ImGui::CollapsingHeader("Display", ImGuiTreeNodeFlags_DefaultOpen)) {
            const char* displayModes[] = {"Rendered", "Base Color", "Normal", "Roughness", "Metalness", "Clearcoat", "Clearcoat Roughness"};
            if (ImGui::Combo("Mode", &mDisplayMode, displayModes, IM_ARRAYSIZE(displayModes))) {
                mUpdateImage();
            }
        }
        if (ImGui::CollapsingHeader("Skybox", ImGuiTreeNodeFlags_DefaultOpen)) {
            const bool hasSkybox = mResourceManager->skyboxIndex() != UINT32_MAX;
            ImGui::AlignTextToFramePadding();
            ImGui::TextDisabled("%s", hasSkybox ? mResourceManager->skyboxName().c_str() : "None");
            ImGui::SameLine();
            if (hasSkybox) {
                if (ImGui::Button("Remove##skybox") && mRemoveSkybox) {
                    mRemoveSkybox();
                }
            } else if (ImGui::Button("Load##skybox")) {
                mSkyboxFileDialog.open(ASSETS_PATH, {".hdr", ".png", ".jpg", ".jpeg", ".bmp", ".tga"});
            }
            if (mSkyboxFileDialog.draw("Select Skybox") && mLoadSkybox) {
                mLoadSkybox(mSkyboxFileDialog.result());
            }
        }
        if (ImGui::CollapsingHeader("Direct Light", ImGuiTreeNodeFlags_DefaultOpen)) {
            if (ImGui::DragFloat3("Direction", &mResourceManager->directLight().dir.x, 0.005f, -1.0f, 1.0f)) {
                mUpdateImage();
            }
            if (ImGui::DragFloat("Intensity", &mResourceManager->directLight().intensity, 0.05f, 0.0f, 10.0f)) {
                mUpdateImage();
            }
        }
        if (ImGui::CollapsingHeader("Performance", ImGuiTreeNodeFlags_DefaultOpen)) {
            if (ImGui::Checkbox("NEE", &mNee)) {
                mUpdateImage();
            }
            if (ImGui::DragInt("SPP", &mSPP, 0.05f, 1, INT_MAX)) {
                mUpdateImage();
            }
            if (ImGui::DragInt("Min Bounces", &mMinDepth, 0.05f, 0, mMaxDepth)) {
                mUpdateImage();
            }
            if (ImGui::DragInt("Max Bounces", &mMaxDepth, 0.05f, 1, INT_MAX)) {
                mUpdateImage();
            }
        }
        if (ImGui::CollapsingHeader("Tonemap", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::Checkbox("ACES", &mTonemap);
            ImGui::DragFloat("Exposure", &mExposure, 0.01f, 0.0f, 100.0f, "%.2f");
        }
        ImGui::Unindent(4.0f);
    }

    void AppUI::drawObjectTab(const AppUIDrawInfo& info) {
        if (!info.selectedInstances || info.selectedInstances->empty()) {
            ImGui::Text("Click a mesh in the viewport");
            return;
        }
        const std::vector<uint32_t>& selected = *info.selectedInstances;

        if (ImGui::Button(ICON_FA_COPY " Duplicate") && mDuplicateInstances) {
            mDuplicateInstances(selected);
            return;
        }
        ImGui::SameLine();
        const bool canDelete = mResourceManager->instances().size() > selected.size();
        ImGui::BeginDisabled(!canDelete);
        const bool deleteClicked = ImGui::Button(ICON_FA_TRASH " Delete");
        ImGui::EndDisabled();
        if (deleteClicked && canDelete && mRemoveInstances) {
            mRemoveInstances(selected);
            return;
        }

        if (selected.size() > 1) {
            ImGui::Separator();
            ImGui::Text("%zu objects selected", selected.size());
            ImGui::TextDisabled("Use the gizmo (T/R) to move or rotate them together.");
            return;
        }

        InstanceData& instance = mResourceManager->instances()[info.activeInstance];
        if (VkImGui::beginGroup(ICON_FA_CIRCLE_INFO " Object")) {
            if (VkImGui::beginCompactTable("##object_status", 6.0f)) {
                VkImGui::row("Name"         , instance.name.c_str());
                VkImGui::row("Mesh Name"    , instance.meshName.c_str());
                VkImGui::row("Material index", std::to_string(instance.materialIndex).c_str());
                VkImGui::endCompactTable();
            }
            VkImGui::endGroup();
        }

        if (VkImGui::beginGroup(ICON_FA_PALETTE " Material")) {
            auto& materials = mResourceManager->materials();
            Material& material = materials[instance.materialIndex];
            std::vector<std::string> materialItems;
            materialItems.reserve(materials.size());
            for (size_t i = 0; i < materials.size(); ++i) {
                materialItems.push_back("#" + std::to_string(i) + " " + materials[i].name);
            }
            if (ImGui::BeginCombo(" ", materialItems[instance.materialIndex].c_str())) {
                for (size_t i = 0; i < materials.size(); ++i) {
                    if (ImGui::Selectable(materialItems[i].c_str())) {
                        instance.materialIndex = i;
                        mResourceManager->updateInstance(info.activeInstance);
                        mUpdateImage();
                    }
                }
                ImGui::EndCombo();
            }
            if (ImGui::CollapsingHeader("Surface", ImGuiTreeNodeFlags_DefaultOpen)) {
                if (VkImGui::colorEdit3("Base Color", material.baseColor)) {
                    mResourceManager->updateMaterial(instance.materialIndex);
                    mUpdateImage();
                }
                if (ImGui::SliderFloat("Metalness", &material.metalness, 0.0f, 1.0f, "%.2f")) {
                    mResourceManager->updateMaterial(instance.materialIndex);
                    mUpdateImage();
                }
                if (ImGui::SliderFloat("Roughness", &material.roughness, 0.0f, 1.0f, "%.2f")) {
                    mResourceManager->updateMaterial(instance.materialIndex);
                    mUpdateImage();
                }
                if (ImGui::SliderFloat("Opacity", &material.opacity, 0.0f, 1.0f, "%.2f")) {
                    mResourceManager->updateMaterial(instance.materialIndex);
                    mUpdateImage();
                }
            }
            if (ImGui::CollapsingHeader("Specular", ImGuiTreeNodeFlags_DefaultOpen)) {
                if (ImGui::SliderFloat("Weight##specular", &material.specular, 0.0f, 1.0f, "%.2f")) {
                    mResourceManager->updateMaterial(instance.materialIndex);
                    mUpdateImage();
                }
            }
            if (ImGui::CollapsingHeader("Transmission", ImGuiTreeNodeFlags_DefaultOpen)) {
                if (ImGui::SliderFloat("Weight##transmission", &material.transmission, 0.0f, 1.0f, "%.2f")) {
                    mResourceManager->updateMaterial(instance.materialIndex);
                    mUpdateImage();
                }
                if (ImGui::SliderFloat("IOR", &material.ior, 1.0f, 3.0f, "%.2f")) {
                    mResourceManager->updateMaterial(instance.materialIndex);
                    mUpdateImage();
                }
                if (VkImGui::colorEdit3("Absorption", material.absorption)) {
                    mResourceManager->updateMaterial(instance.materialIndex);
                    mUpdateImage();
                }
            }
            if (ImGui::CollapsingHeader("Coating", ImGuiTreeNodeFlags_DefaultOpen)) {
                if (ImGui::SliderFloat("Weight##coating", &material.clearcoat, 0.0f, 1.0f, "%.2f")) {
                    mResourceManager->updateMaterial(instance.materialIndex);
                    mUpdateImage();
                }
                if (ImGui::SliderFloat("Roughness##coating", &material.clearcoatRoughness, 0.0f, 1.0f, "%.2f")) {
                    mResourceManager->updateMaterial(instance.materialIndex);
                    mUpdateImage();
                }
            }
            if (ImGui::CollapsingHeader("Emission", ImGuiTreeNodeFlags_DefaultOpen)) {
                if (ImGui::DragFloat("Luminance", &material.luminance, 0.05f, 0.0f, 100.0f)) {
                    mResourceManager->updateMaterial(instance.materialIndex);
                    mUpdateImage();
                }
            }
            if (ImGui::CollapsingHeader("Textures", ImGuiTreeNodeFlags_DefaultOpen)) {
                if (VkImGui::beginCompactTable("##textures", 6.0f)) {
                    const std::vector<std::string> extensions =
                        {".png", ".jpg", ".jpeg", ".bmp", ".tga", ".hdr", ".exr", ".ktx", ".dds"};

                    ImGui::TableNextRow();
                    ImGui::TableSetColumnIndex(0);
                    ImGui::AlignTextToFramePadding();
                    ImGui::TextDisabled("%s", "Base Color");
                    const bool hasBaseColor = material.baseColorTexIndex != UINT32_MAX;
                    ImGui::TableSetColumnIndex(1);
                    ImGui::AlignTextToFramePadding();
                    ImGui::TextDisabled("%s", hasBaseColor ? material.baseColorTexName.c_str() : "None");
                    ImGui::TableSetColumnIndex(2);
                    ImGui::AlignTextToFramePadding();
                    if (hasBaseColor) {
                        if (ImGui::Button("Clear##basecolor")) {
                            material.baseColorTexIndex = UINT32_MAX;
                            material.baseColorTexName.clear();
                            mResourceManager->updateMaterial(instance.materialIndex);
                            mUpdateImage();
                        }
                    } else if (ImGui::Button("Upload##basecolor")) {
                        mUploadTextureType = 0;
                        mFileDialog.open(ASSETS_PATH, extensions);
                    }

                    ImGui::TableNextRow();
                    ImGui::TableSetColumnIndex(0);
                    ImGui::AlignTextToFramePadding();
                    ImGui::TextDisabled("%s", "Normal");
                    const bool hasNormal = material.normalTexIndex != UINT32_MAX;
                    ImGui::TableSetColumnIndex(1);
                    ImGui::AlignTextToFramePadding();
                    ImGui::TextDisabled("%s", hasNormal ? material.normalTexName.c_str() : "None");
                    ImGui::TableSetColumnIndex(2);
                    ImGui::AlignTextToFramePadding();
                    if (hasNormal) {
                        if (ImGui::Button("Clear##normal")) {
                            material.normalTexIndex = UINT32_MAX;
                            material.normalTexName.clear();
                            mResourceManager->updateMaterial(instance.materialIndex);
                            mUpdateImage();
                        }
                    } else if (ImGui::Button("Upload##normal")) {
                        mUploadTextureType = 1;
                        mFileDialog.open(ASSETS_PATH, extensions);
                    }

                    ImGui::TableNextRow();
                    ImGui::TableSetColumnIndex(0);
                    ImGui::AlignTextToFramePadding();
                    ImGui::TextDisabled("%s", "Metal/Rough");
                    const bool hasMetalRough = material.metalRoughnessTexIndex != UINT32_MAX;
                    ImGui::TableSetColumnIndex(1);
                    ImGui::AlignTextToFramePadding();
                    ImGui::TextDisabled("%s", hasMetalRough ? material.metalRoughnessTexName.c_str() : "None");
                    ImGui::TableSetColumnIndex(2);
                    ImGui::AlignTextToFramePadding();
                    if (hasMetalRough) {
                        if (ImGui::Button("Clear##metalrough")) {
                            material.metalRoughnessTexIndex = UINT32_MAX;
                            material.metalRoughnessTexName.clear();
                            mResourceManager->updateMaterial(instance.materialIndex);
                            mUpdateImage();
                        }
                    } else if (ImGui::Button("Upload##metalrough")) {
                        mUploadTextureType = 2;
                        mFileDialog.open(ASSETS_PATH, extensions);
                    }

                    ImGui::TableNextRow();
                    ImGui::TableSetColumnIndex(0);
                    ImGui::AlignTextToFramePadding();
                    ImGui::TextDisabled("%s", "Clearcoat");
                    const bool hasClearcoat = material.clearcoatTexIndex != UINT32_MAX;
                    ImGui::TableSetColumnIndex(1);
                    ImGui::AlignTextToFramePadding();
                    ImGui::TextDisabled("%s", hasClearcoat ? material.clearcoatTexName.c_str() : "None");
                    ImGui::TableSetColumnIndex(2);
                    ImGui::AlignTextToFramePadding();
                    if (hasClearcoat) {
                        if (ImGui::Button("Clear##clearcoat")) {
                            material.clearcoatTexIndex = UINT32_MAX;
                            material.clearcoatTexName.clear();
                            mResourceManager->updateMaterial(instance.materialIndex);
                            mUpdateImage();
                        }
                    } else if (ImGui::Button("Upload##clearcoat")) {
                        mUploadTextureType = 3;
                        mFileDialog.open(ASSETS_PATH, extensions);
                    }

                    ImGui::TableNextRow();
                    ImGui::TableSetColumnIndex(0);
                    ImGui::AlignTextToFramePadding();
                    ImGui::TextDisabled("%s", "Clearcoat Rough");
                    const bool hasClearcoatRough = material.clearcoatRoughnessTexIndex != UINT32_MAX;
                    ImGui::TableSetColumnIndex(1);
                    ImGui::AlignTextToFramePadding();
                    ImGui::TextDisabled("%s", hasClearcoatRough ? material.clearcoatRoughnessTexName.c_str() : "None");
                    ImGui::TableSetColumnIndex(2);
                    ImGui::AlignTextToFramePadding();
                    if (hasClearcoatRough) {
                        if (ImGui::Button("Clear##clearcoatrough")) {
                            material.clearcoatRoughnessTexIndex = UINT32_MAX;
                            material.clearcoatRoughnessTexName.clear();
                            mResourceManager->updateMaterial(instance.materialIndex);
                            mUpdateImage();
                        }
                    } else if (ImGui::Button("Upload##clearcoatrough")) {
                        mUploadTextureType = 4;
                        mFileDialog.open(ASSETS_PATH, extensions);
                    }

                    if (mFileDialog.draw("Select Texture")) {
                        mUploadTexture(mFileDialog.result(), instance.materialIndex, mUploadTextureType);
                        mUpdateImage();
                    }
                    VkImGui::endCompactTable();
                }
            }
        }

        if (VkImGui::beginGroup(ICON_FA_ARROWS_UP_DOWN_LEFT_RIGHT " Transform")) {
            bool changed = false;
            Transform& transform = instance.transform;
            if (ImGui::DragFloat3("Position", &transform.position[0], 0.1f))
                changed = true;

            static glm::vec3 uiRotation{};
            static glm::quat cachedRotation{};
            static uint32_t cachedInstance = UINT32_MAX;
            if (info.activeInstance != cachedInstance ||
                instance.transform.rotation != cachedRotation) {
                uiRotation = glm::degrees(glm::eulerAngles(instance.transform.rotation));
                cachedInstance = info.activeInstance;
            }
            if (ImGui::DragFloat3("Rotation", &uiRotation[0], 0.5f)) {
                uiRotation = clampRotation(uiRotation);
                transform.rotation = glm::normalize(glm::quat(glm::radians(uiRotation)));
                changed = true;
            }
            cachedRotation = instance.transform.rotation;

            if (ImGui::DragFloat3("Scale", &transform.scale[0], 0.05f))
                changed = true;

            if (changed) {
                mResourceManager->updateInstanceTransform(info.activeInstance);
                mUpdateImage();
            }
            VkImGui::endGroup();
        }
    }

    void AppUI::drawSettings(const AppUIDrawInfo& info) {
        if (ImGui::Begin("Settings")) {
            const TabPanel panel = {
                {ICON_FA_CUBE   " Object", 0, [this, info]{drawObjectTab(info);}},
                {ICON_FA_IMAGES " Render", 1, [this, info]{drawRenderTab(info);}},
                {ICON_FA_VIDEO  " Camera", 2, [this, info]{drawCameraTab(info);}},
            };
            static uint32_t activeSettingsTabIndex = 0;
            VkImGui::tabPanel(panel, activeSettingsTabIndex);
        }
        ImGui::End();
    }
}