//
// Created by igor on 6/12/26.
//

#include "View/AppUI.hpp"
#include "IconsFontAwesome6.h"
#include <ImGuizmo.h>
#include <algorithm>
#include <filesystem>
#include <cstring>
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
    mContext(info.context), mSwapchain(info.swapchain),
    mSettings(info.renderSettings), mCommands(info.commands), mScene(info.scene) {
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
        if (info.drawUI) {
            drawOverView(info);
            drawScene(info);
            drawSettings(info);
        } else {
            drawCursorDot();
        }
        if (mUpdateImage) {
            push(CommandType::UPDATE_IMAGE);
            mUpdateImage = false;
        }
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
            if (b.x - a.x > 3.0f && b.y - a.y > 3.0f) {
                const ImVec2 scale = io.DisplayFramebufferScale;
                push(CommandType::REGION_SELECT, RegionSelectPayload{
                    static_cast<int>(a.x * scale.x), static_cast<int>(a.y * scale.y),
                    static_cast<int>(b.x * scale.x), static_cast<int>(b.y * scale.y), io.KeyShift});
            }
        }
    }

    void AppUI::drawCursorDot() {
        if (!ImGui::IsMousePosValid()) return;
        ImGui::SetMouseCursor(ImGuiMouseCursor_None);
        ImGui::GetForegroundDrawList()->AddCircleFilled(ImGui::GetIO().MousePos, 3.0f, IM_COL32(95, 40, 120, 255));
    }

    void AppUI::drawGizmo(const AppUIDrawInfo& info) {
        if (info.activeInstance >= mScene->instances().size()) return;
        if (!ImGui::GetIO().WantTextInput) {
            if (ImGui::IsKeyPressed(ImGuiKey_T)) mGizmoOp = ImGuizmo::TRANSLATE;
            if (ImGui::IsKeyPressed(ImGuiKey_R)) mGizmoOp = ImGuizmo::ROTATE;
        }

        const glm::mat4& pivot = mScene->instances()[info.activeInstance].world;
        glm::mat4 view  = info.camera->viewMatrix();
        glm::mat4 proj  = info.camera->projectionMatrix();
        proj[1][1] *= -1.0f;
        const glm::vec3 pivotPos = glm::vec3(pivot[3]);
        glm::mat3 pivotBasis(pivot);
        pivotBasis[0] = glm::normalize(pivotBasis[0]);
        pivotBasis[1] = glm::normalize(pivotBasis[1]);
        pivotBasis[2] = glm::normalize(pivotBasis[2]);
        const glm::quat pivotRot = glm::normalize(glm::quat_cast(pivotBasis));
        glm::mat4 model = glm::translate(glm::mat4(1.0f), pivotPos) * glm::toMat4(pivotRot);

        const ImVec2 displaySize = ImGui::GetIO().DisplaySize;
        const ImGuizmo::MODE mode = mGizmoOp == ImGuizmo::ROTATE ? ImGuizmo::LOCAL : ImGuizmo::WORLD;
        ImGuizmo::SetOrthographic(false);
        ImGuizmo::SetDrawlist(ImGui::GetForegroundDrawList());
        ImGuizmo::SetRect(0.0f, 0.0f, displaySize.x, displaySize.y);

        glm::mat4 delta(1.0f);
        if (ImGuizmo::Manipulate(glm::value_ptr(view), glm::value_ptr(proj),
            mGizmoOp, mode, glm::value_ptr(model), glm::value_ptr(delta))) {
            push(CommandType::TRANSFORM_INSTANCES, TransformInstancesPayload{*info.selectedInstances, delta});
        }
    }

    void AppUI::drawOverView(const AppUIDrawInfo& info) {
        if (ImGui::Begin("Overview", nullptr, ImGuiWindowFlags_MenuBar)) {
            if (ImGui::BeginMenuBar()) {
                if (ImGui::BeginMenu("File")) {
                    if (ImGui::MenuItem(ICON_FA_PLUS " Add Model")) {
                        mModelFileDialog.open(ASSETS_PATH, {".glb", ".gltf", ".obj", ".fbx", ".ply", ".dae", ".stl"});
                    }
                    ImGui::Separator();
                    if (ImGui::MenuItem(ICON_FA_CAMERA " Save Image")) {
                        push(CommandType::SAVE_IMAGE);
                    }
                    if (ImGui::MenuItem(ICON_FA_FLOPPY_DISK " Save Scene")) {
                        push(CommandType::SAVE_SCENE);
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
                std::string accumulation = std::format("{:1}", (info.frameCount + 1) * mSettings->spp);
                std::string spp = std::to_string(mSettings->spp);

                if (VkImGui::beginCompactTable("##monitor_status", 2.0f)) {
                    VkImGui::row("FPS"         , fps.c_str());
                    VkImGui::row("Render Time" , renderTime.c_str());
                    VkImGui::row("SPP"         , spp.c_str());
                    VkImGui::row("Accumulation", accumulation.c_str());
                    VkImGui::endCompactTable();
                }
                VkImGui::endGroup();
            }

            if (VkImGui::beginGroup(ICON_FA_MICROCHIP " System")) {
                auto properties = mContext->physicalDeviceProperties();
                auto [width, height] = mSwapchain->extent();
                const auto scale = info.renderScale;
                std::string viewport = std::format("{:1}x{:2}", width, height);
                std::string render = std::format("{:1}x{:2}", (width + scale - 1) / scale, (height + scale - 1) / scale);
                if (VkImGui::beginCompactTable("##monitor_system", 2.0f)) {
                    VkImGui::row("GPU"     , properties.deviceName);
                    VkImGui::row("Viewport", viewport.c_str());
                    VkImGui::row("Render"  , render.c_str());
                    VkImGui::endCompactTable();
                }
                VkImGui::endGroup();
            }

            if (mModelFileDialog.draw("Add Model")) {
                push(CommandType::ADD_MODEL, ModelImportPayload{mModelFileDialog.result()});
            }
        }
        ImGui::End();
    }

    void AppUI::drawScene(const AppUIDrawInfo& info) {
        if (ImGui::Begin(ICON_FA_CUBES " Scene")) {
            const auto& instances = mScene->instances();
            std::vector<std::vector<uint32_t>> children(instances.size());
            std::vector<uint32_t> roots;
            for (uint32_t i = 0; i < instances.size(); ++i) {
                const int32_t parent = instances[i].parentIndex;
                if (parent >= 0 && parent < static_cast<int32_t>(instances.size()))
                    children[parent].push_back(i);
                else
                    roots.push_back(i);
            }
            const auto byName = [&](const uint32_t a, const uint32_t b) {
                return instances[a].name < instances[b].name;
            };
            std::sort(roots.begin(), roots.end(), byName);
            for (auto& siblings : children) std::sort(siblings.begin(), siblings.end(), byName);

            ImGui::TextDisabled("%zu instances", instances.size());
            ImGui::BeginChild("##scene_list", ImVec2(0, 0), ImGuiChildFlags_Borders);
            for (const uint32_t root : roots) drawInstanceNode(root, children, info);
            ImGui::EndChild();
        }
        ImGui::End();
    }

    void AppUI::drawInstanceNode(const uint32_t index, const std::vector<std::vector<uint32_t>>& children,
                                 const AppUIDrawInfo& info) {
        const InstanceData& instance = mScene->instances()[index];
        const std::vector<uint32_t>* selected = info.selectedInstances;
        const bool isSelected = selected &&
            std::find(selected->begin(), selected->end(), index) != selected->end();

        const std::string& name = !instance.name.empty() ? instance.name : instance.meshName;
        const std::string label = (name.empty() ? "Node" : name) + "##inst" + std::to_string(index);

        ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_SpanAvailWidth;
        if (isSelected) flags |= ImGuiTreeNodeFlags_Selected;
        const bool hasChildren = !children[index].empty();
        if (!hasChildren) flags |= ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen;

        const bool open = ImGui::TreeNodeEx(label.c_str(), flags);
        if (ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen()) {
            push(CommandType::SELECT_INSTANCE,
                 SelectInstancePayload{index, ImGui::GetIO().KeyCtrl || ImGui::GetIO().KeyShift});
        }
        if (open && hasChildren) {
            for (const uint32_t child : children[index]) drawInstanceNode(child, children, info);
            ImGui::TreePop();
        }
    }

    void AppUI::drawCameraTab(const AppUIDrawInfo &info) {
        glm::vec3 position = info.camera->position();
        if (ImGui::DragFloat3("Position", &position.x, 0.05f, -FLT_MAX, FLT_MAX)) {
            info.camera->setPosition(position);
            mUpdateImage = true;
        }
        float fov = info.camera->FOV();
        if (ImGui::SliderFloat("FOV", &fov, 10, 140, "%.2f deg")) {
            info.camera->zoom(info.camera->FOV() - fov);
            mUpdateImage = true;
        }
        const bool isFlyCamera = info.camera->type() == cs::CameraType::FLY;
        if (VkImGui::selectableButton("Fly", isFlyCamera)) {
            push(CommandType::SET_CAMERA_FLY);
        }
        ImGui::SameLine(0.0f, 5.0f);
        if (VkImGui::selectableButton("Orbital", !isFlyCamera)) {
            push(CommandType::SET_CAMERA_ORBITAL);
            mUpdateImage = true;
        }
        ImGui::SameLine();
        ImGui::Text("Type");
    }

    void AppUI::drawRenderTab(const AppUIDrawInfo& info) {
        ImGui::Indent(4.0f);
        if (ImGui::CollapsingHeader("Display", ImGuiTreeNodeFlags_DefaultOpen)) {
            const char* displayModes[] = {"Rendered", "Base Color", "Normal", "Roughness", "Metalness", "Clearcoat", "Clearcoat Roughness"};
            if (ImGui::Combo("Mode", &mSettings->displayMode, displayModes, IM_ARRAYSIZE(displayModes))) {
                mUpdateImage = true;
            }
        }
        if (ImGui::CollapsingHeader("Skybox", ImGuiTreeNodeFlags_DefaultOpen)) {
            const bool hasSkybox = mScene->skyboxIndex() != UINT32_MAX;
            ImGui::AlignTextToFramePadding();
            const std::string skyboxName = hasSkybox
                ? std::filesystem::path(mScene->skyboxPath()).filename().string() : "None";
            ImGui::TextDisabled("%s", skyboxName.c_str());
            ImGui::SameLine();
            if (hasSkybox) {
                if (ImGui::Button("Remove##skybox")) {
                    push(CommandType::REMOVE_SKYBOX);
                }
            } else if (ImGui::Button("Load##skybox")) {
                mSkyboxFileDialog.open(ASSETS_PATH, {".hdr", ".png", ".jpg", ".jpeg", ".bmp", ".tga"});
            }
            if (mSkyboxFileDialog.draw("Select Skybox")) {
                push(CommandType::LOAD_SKYBOX, SkyboxPayload{mSkyboxFileDialog.result()});
            }
            if (!hasSkybox) {
                glm::vec3 skyColor = mScene->skyColor();
                if (VkImGui::colorEdit3("Sky Color", skyColor))
                    push(CommandType::SET_SKY_COLOR, SkyColorPayload{skyColor});
            }
        }
        if (ImGui::CollapsingHeader("Direct Light", ImGuiTreeNodeFlags_DefaultOpen)) {
            DirectLight light = mScene->directLight();
            bool changed = false;
            changed |= ImGui::DragFloat3("Direction", &light.dir.x, 0.005f, -1.0f, 1.0f);
            changed |= ImGui::DragFloat("Intensity", &light.intensity, 0.05f, 0.0f, 10.0f);
            if (changed)
                push(CommandType::SET_DIRECT_LIGHT, DirectLightPayload{light});
        }
        if (ImGui::CollapsingHeader("NEE", ImGuiTreeNodeFlags_DefaultOpen)) {
            if (ImGui::Checkbox("Light sources", &mSettings->nee)) {
                mUpdateImage = true;
            }
            ImGui::BeginDisabled(mScene->skyboxIndex() == UINT32_MAX);
            if (ImGui::Checkbox("Environment", &mSettings->envNee)) {
                mUpdateImage = true;
            }
            ImGui::EndDisabled();
        }
        if (ImGui::CollapsingHeader("Resolution", ImGuiTreeNodeFlags_DefaultOpen)) {
            if (ImGui::DragInt("Render Scale", &mSettings->renderScale, 0.05f, 1, 16)) {
                mUpdateImage = true;
            }
            ImGui::DragInt("Motion Scale", &mSettings->motionScale, 0.05f, mSettings->renderScale, 16);
        }
        if (ImGui::CollapsingHeader("Performance", ImGuiTreeNodeFlags_DefaultOpen)) {
            if (ImGui::DragInt("SPP", &mSettings->spp, 0.05f, 1, INT_MAX)) {
                mUpdateImage = true;
            }
            if (ImGui::DragInt("Min Bounces", &mSettings->minDepth, 0.05f, 0, mSettings->maxDepth)) {
                mUpdateImage = true;
            }
            if (ImGui::DragInt("Max Bounces", &mSettings->maxDepth, 0.05f, 1, INT_MAX)) {
                mUpdateImage = true;
            }
        }
        if (ImGui::CollapsingHeader("Depth of Field", ImGuiTreeNodeFlags_DefaultOpen)) {
            if (ImGui::DragFloat("Aperture", &mSettings->aperture, 0.001f, 0.0f, 5.0f, "%.3f")) {
                mUpdateImage = true;
            }
            if (ImGui::DragFloat("Focus Distance", &mSettings->focusDistance, 0.05f, 0.01f, 1000.0f, "%.2f")) {
                mUpdateImage = true;
            }
        }
        if (ImGui::CollapsingHeader("Tonemap", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::Checkbox("ACES", &mSettings->tonemap);
            ImGui::DragFloat("Exposure", &mSettings->exposure, 0.01f, 0.0f, 100.0f, "%.2f");
        }
        ImGui::Unindent(4.0f);
    }

    void AppUI::drawObjectTab(const AppUIDrawInfo& info) {
        if (!info.selectedInstances || info.selectedInstances->empty()) {
            ImGui::Text("Click a mesh in the viewport");
            return;
        }
        const std::vector<uint32_t>& selected = *info.selectedInstances;

        if (ImGui::Button(ICON_FA_COPY " Duplicate")) {
            push(CommandType::DUPLICATE_INSTANCES, InstancesPayload{selected});
            return;
        }
        ImGui::SameLine();
        const std::vector<bool> doomed = mScene->subtreeMask(selected);
        const auto removed = static_cast<size_t>(std::count(doomed.begin(), doomed.end(), true));
        const bool deleteClicked = ImGui::Button(ICON_FA_TRASH " Delete");
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip(removed == mScene->instances().size()
                ? "Removes all %zu instances" : "Removes %zu instances", removed);
        if (deleteClicked) {
            push(CommandType::REMOVE_INSTANCES, InstancesPayload{selected});
            return;
        }

        if (selected.size() > 1) ImGui::TextDisabled("%zu objects selected", selected.size());
        else ImGui::TextDisabled("Instance selected");

        const uint32_t active = info.activeInstance;
        if (active >= mScene->instances().size()) {
            ImGui::Text("Selection is out of date");
            return;
        }
        const InstanceData& instance = mScene->instances()[active];
        const bool showMaterial = !instance.isGroup() && selected.size() == 1;
        if (VkImGui::beginGroup(ICON_FA_CIRCLE_INFO " Object")) {
            char nameBuffer[128]{};
            std::strncpy(nameBuffer, instance.name.c_str(), sizeof(nameBuffer) - 1);
            if (ImGui::InputText("Name##instance", nameBuffer, sizeof(nameBuffer)))
                push(CommandType::SET_INSTANCE_NAME, SetInstanceNamePayload{active, nameBuffer});
            if (showMaterial && VkImGui::beginCompactTable("##object_status", 6.0f)) {
                VkImGui::row("Mesh Name"    , instance.meshName.c_str());
                VkImGui::row("Material index", std::to_string(instance.materialIndex).c_str());
                VkImGui::endCompactTable();
            }
            VkImGui::endGroup();
        }

        if (showMaterial && VkImGui::beginGroup(ICON_FA_PALETTE " Material")) {
            const auto& materials = mScene->materials();
            const Material& material = materials[instance.materialIndex];
            std::vector<std::string> materialItems;
            materialItems.reserve(materials.size());
            for (size_t i = 0; i < materials.size(); ++i) {
                materialItems.push_back("#" + std::to_string(i) + " " + materials[i].name);
            }
            if (ImGui::BeginCombo(" ", materialItems[instance.materialIndex].c_str())) {
                for (size_t i = 0; i < materials.size(); ++i) {
                    if (ImGui::Selectable(materialItems[i].c_str())) {
                        push(CommandType::SET_INSTANCE_MATERIAL,
                             SetInstanceMaterialPayload{active, static_cast<uint32_t>(i)});
                    }
                }
                ImGui::EndCombo();
            }
            ImGui::SameLine();
            if (ImGui::Button(ICON_FA_PLUS)) {
                push(CommandType::ADD_MATERIAL, MaterialPayload{active});
                VkImGui::endGroup();
                return;
            }

            Material edited = material;
            bool changed = false;

            char nameBuffer[128]{};
            std::strncpy(nameBuffer, material.name.c_str(), sizeof(nameBuffer) - 1);
            if (ImGui::InputText("Name##material", nameBuffer, sizeof(nameBuffer))) {
                edited.name = nameBuffer;
                changed = true;
            }
            if (ImGui::CollapsingHeader("Surface", ImGuiTreeNodeFlags_DefaultOpen)) {
                changed |= VkImGui::colorEdit3("Base Color", edited.baseColor);
                changed |= ImGui::SliderFloat("Metalness", &edited.metalness, 0.0f, 1.0f, "%.2f");
                changed |= ImGui::SliderFloat("Roughness", &edited.roughness, 0.0f, 1.0f, "%.2f");
                changed |= ImGui::SliderFloat("Anisotropy", &edited.anisotropy, 0.0f, 1.0f, "%.2f");
                changed |= ImGui::SliderFloat("Opacity", &edited.opacity, 0.0f, 1.0f, "%.2f");
            }
            if (ImGui::CollapsingHeader("Specular", ImGuiTreeNodeFlags_DefaultOpen)) {
                changed |= ImGui::SliderFloat("Weight##specular", &edited.specular, 0.0f, 1.0f, "%.2f");
            }
            if (ImGui::CollapsingHeader("Transmission", ImGuiTreeNodeFlags_DefaultOpen)) {
                changed |= ImGui::SliderFloat("Weight##transmission", &edited.transmission, 0.0f, 1.0f, "%.2f");
                changed |= ImGui::SliderFloat("IOR", &edited.ior, 1.0f, 3.0f, "%.2f");
                changed |= VkImGui::colorEdit3("Absorption", edited.absorption);
                bool thin = edited.thin > 0.5f;
                if (ImGui::Checkbox("Thin (flat glass)", &thin)) { edited.thin = thin ? 1.0f : 0.0f; changed = true; }
            }
            if (ImGui::CollapsingHeader("Coating", ImGuiTreeNodeFlags_DefaultOpen)) {
                changed |= ImGui::SliderFloat("Weight##coating", &edited.clearcoat, 0.0f, 1.0f, "%.2f");
                changed |= ImGui::SliderFloat("Roughness##coating", &edited.clearcoatRoughness, 0.0f, 1.0f, "%.2f");
            }
            if (ImGui::CollapsingHeader("Emission", ImGuiTreeNodeFlags_DefaultOpen)) {
                changed |= ImGui::DragFloat("Luminance", &edited.luminance, 0.05f, 0.0f, 100.0f);
            }
            if (ImGui::CollapsingHeader("Textures", ImGuiTreeNodeFlags_DefaultOpen)) {
                if (VkImGui::beginCompactTable("##textures", 6.0f)) {
                    const std::vector<std::string> extensions =
                        {".png", ".jpg", ".jpeg", ".bmp", ".tga", ".hdr", ".exr", ".ktx", ".dds"};

                    const auto textureRow = [&](const char* label, const char* id, int type,
                                                const uint32_t texIndex, const std::string& texPath,
                                                uint32_t& editedIndex, std::string& editedPath) {
                        ImGui::TableNextRow();
                        ImGui::TableSetColumnIndex(0);
                        ImGui::AlignTextToFramePadding();
                        ImGui::TextDisabled("%s", label);
                        const bool has = texIndex != UINT32_MAX;
                        ImGui::TableSetColumnIndex(1);
                        ImGui::AlignTextToFramePadding();
                        const std::string texName = has ? std::filesystem::path(texPath).filename().string() : "None";
                        ImGui::TextDisabled("%s", texName.c_str());
                        ImGui::TableSetColumnIndex(2);
                        ImGui::AlignTextToFramePadding();
                        if (has) {
                            if (ImGui::Button((std::string("Clear##") + id).c_str())) {
                                editedIndex = UINT32_MAX;
                                editedPath.clear();
                                changed = true;
                            }
                        } else if (ImGui::Button((std::string("Upload##") + id).c_str())) {
                            mUploadTextureType = type;
                            mFileDialog.open(ASSETS_PATH, extensions);
                        }
                    };

                    textureRow("Base Color", "basecolor", 0, material.baseColorTexIndex, material.baseColorTexPath,
                               edited.baseColorTexIndex, edited.baseColorTexPath);
                    textureRow("Normal", "normal", 1, material.normalTexIndex, material.normalTexPath,
                               edited.normalTexIndex, edited.normalTexPath);

                    ImGui::TableNextRow();
                    ImGui::TableSetColumnIndex(0);
                    ImGui::AlignTextToFramePadding();
                    ImGui::TextDisabled("%s", "Normal Scale");
                    ImGui::TableSetColumnIndex(1);
                    ImGui::SetNextItemWidth(-FLT_MIN);
                    changed |= ImGui::SliderFloat("##normalScale", &edited.normalScale, 0.0f, 2.0f, "%.2f");

                    textureRow("Metal/Rough", "metalrough", 2, material.metalRoughnessTexIndex, material.metalRoughnessTexPath,
                               edited.metalRoughnessTexIndex, edited.metalRoughnessTexPath);
                    textureRow("Clearcoat", "clearcoat", 3, material.clearcoatTexIndex, material.clearcoatTexPath,
                               edited.clearcoatTexIndex, edited.clearcoatTexPath);
                    textureRow("Clearcoat Rough", "clearcoatrough", 4, material.clearcoatRoughnessTexIndex, material.clearcoatRoughnessTexPath,
                               edited.clearcoatRoughnessTexIndex, edited.clearcoatRoughnessTexPath);

                    if (mFileDialog.draw("Select Texture")) {
                        push(CommandType::UPLOAD_TEXTURE, UploadTexturePayload{mFileDialog.result(), instance.materialIndex, mUploadTextureType});
                        mUpdateImage = true;
                    }
                    VkImGui::endCompactTable();
                }
            }

            if (changed) push(CommandType::SET_MATERIAL, SetMaterialPayload{instance.materialIndex, edited});
        }

        if (VkImGui::beginGroup(ICON_FA_ARROWS_UP_DOWN_LEFT_RIGHT " Transform")) {
            bool changed = false;
            Transform edited = instance.transform;
            if (ImGui::DragFloat3("Position", &edited.position[0], 0.1f))
                changed = true;

            static glm::vec3 uiRotation{};
            static glm::quat cachedRotation{};
            static uint32_t cachedInstance = UINT32_MAX;
            if (active != cachedInstance ||
                std::abs(glm::dot(instance.transform.rotation, cachedRotation)) < 0.99999f) {
                uiRotation = glm::degrees(glm::eulerAngles(instance.transform.rotation));
                cachedInstance = active;
            }
            if (ImGui::DragFloat3("Rotation", &uiRotation[0], 0.5f)) {
                uiRotation = clampRotation(uiRotation);
                edited.rotation = glm::normalize(glm::quat(glm::radians(uiRotation)));
                changed = true;
            }
            cachedRotation = edited.rotation;

            if (ImGui::DragFloat3("Scale", &edited.scale[0], 0.05f))
                changed = true;

            if (changed) {
                if (selected.size() > 1) {
                    const std::vector<InstanceData>& instances = mScene->instances();
                    const glm::mat4 parentWorld = instance.parentIndex >= 0
                        ? instances[instance.parentIndex].world : glm::mat4(1.0f);
                    const glm::mat4 localDelta = edited.matrix() * glm::inverse(instance.transform.matrix());
                    const glm::mat4 delta = parentWorld * localDelta * glm::inverse(parentWorld);
                    push(CommandType::TRANSFORM_INSTANCES, TransformInstancesPayload{selected, delta});
                } else {
                    push(CommandType::SET_INSTANCE_TRANSFORM, SetInstanceTransformPayload{active, edited});
                }
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