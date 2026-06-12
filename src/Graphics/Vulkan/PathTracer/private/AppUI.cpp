//
// Created by igor on 6/12/26.
//

#include "AppUI.hpp"
#include "IconsFontAwesome6.h"

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
        if (!info.drawUI) {
            mImGui.endFrame();
            return;
        }
        drawOverView(info);
        drawSettings(info);
        mImGui.endFrame();
    }

    void AppUI::drawOverView(const AppUIDrawInfo& info) {
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
        if (ImGui::CollapsingHeader("Direct Light", ImGuiTreeNodeFlags_DefaultOpen)) {
            if (ImGui::DragFloat3("Direction", &mResourceManager->directLight().dir.x, 0.005f, -1.0f, 1.0f)) {
                mUpdateImage();
            }
            if (ImGui::DragFloat("Intensity", &mResourceManager->directLight().intensity, 0.05f, 0.0f, 10.0f)) {
                mUpdateImage();
            }
        }
        if (ImGui::CollapsingHeader("Performance", ImGuiTreeNodeFlags_DefaultOpen)) {
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
        ImGui::Unindent(4.0f);
    }

    void AppUI::drawObjectTab(const AppUIDrawInfo& info) {
        if (info.selectedInstanceIndex == 0) {
            ImGui::Text("Click a mesh in the viewport");
            return;
        }
        InstanceData& instance = mResourceManager->instances()[info.selectedInstanceIndex - 1];
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
                        mResourceManager->updateInstance(info.selectedInstanceIndex - 1);
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
            }
            if (ImGui::CollapsingHeader("Textures", ImGuiTreeNodeFlags_DefaultOpen)) {
                if (VkImGui::beginCompactTable("##textures", 6.0f)) {
                    ImGui::TableNextRow();
                    ImGui::TableSetColumnIndex(0);
                    ImGui::AlignTextToFramePadding();
                    ImGui::TextDisabled("%s", "Base Color");
                    ImGui::TableSetColumnIndex(1);
                    ImGui::AlignTextToFramePadding();
                    ImGui::TextDisabled("%s", "None");
                    ImGui::TableSetColumnIndex(2);
                    ImGui::AlignTextToFramePadding();
                    if (ImGui::Button("Upload"))
                    {
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
                mResourceManager->updateInstanceTransform(info.selectedInstanceIndex - 1);
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