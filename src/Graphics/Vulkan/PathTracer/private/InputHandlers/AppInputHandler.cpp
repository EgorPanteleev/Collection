//
// Created by igor on 6/12/26.
//

#include "InputHandlers/AppInputHandler.hpp"
#include "PathTracerApp.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>

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
        Model& model = app->mModel;
        switch (command.type) {
            case CommandType::SET_CAMERA_FLY:     model.setActiveCamera(scene::CameraType::FLY);     break;
            case CommandType::SET_CAMERA_ORBITAL: model.setActiveCamera(scene::CameraType::ORBITAL); break;

            case CommandType::PICK_OBJECT: {
                const bool additive = app->mInput.isPressed(Key::LEFT_SHIFT) || app->mInput.isPressed(Key::RIGHT_SHIFT);
                app->mRenderer->pick(app->mInput.cursorPos(), additive);
                break;
            }
            case CommandType::CLEAR_SELECTION: model.clearSelection(); break;
            case CommandType::SELECT_INSTANCE: {
                const auto& p = std::get<SelectInstancePayload>(command.payload);
                model.selectInstance(p.index, p.additive);
                break;
            }
            case CommandType::REGION_SELECT: {
                const auto& p = std::get<RegionSelectPayload>(command.payload);
                const auto [width, height] = app->mRenderer->extent();
                model.regionSelect(p.x0, p.y0, p.x1, p.y1, p.additive, width, height);
                break;
            }
            case CommandType::DUPLICATE_INSTANCES:
                model.duplicateInstances(std::get<InstancesPayload>(command.payload).indices);
                break;
            case CommandType::REMOVE_INSTANCES:
                model.removeInstances(std::get<InstancesPayload>(command.payload).indices);
                break;
            case CommandType::ADD_MATERIAL:
                model.addMaterial(std::get<MaterialPayload>(command.payload).instanceIndex);
                break;
            case CommandType::UPLOAD_TEXTURE: {
                const auto& p = std::get<UploadTexturePayload>(command.payload);
                model.addTexture(p.path, p.materialIndex, p.textureType);
                break;
            }
            case CommandType::LOAD_SKYBOX:
                model.loadSkybox(std::get<SkyboxPayload>(command.payload).path);
                break;
            case CommandType::REMOVE_SKYBOX: model.removeSkybox(); break;
            case CommandType::UPDATE_INSTANCE:
                for (const uint32_t index : std::get<InstancesPayload>(command.payload).indices)
                    model.markInstanceDirty(index);
                break;
            case CommandType::UPDATE_INSTANCE_DATA:
                model.markInstanceDataDirty(std::get<IndexPayload>(command.payload).index);
                break;
            case CommandType::UPDATE_MATERIAL:
                model.markMaterialDirty(std::get<IndexPayload>(command.payload).index);
                break;
            case CommandType::UPDATE_IMAGE: model.requestReset(); break;

            case CommandType::TOGGLE_CONTROL_PANEL: app->mRenderer->toggleUI(); break;

            case CommandType::QUIT:       app->window().close();  break;
            case CommandType::SAVE_IMAGE: app->mRenderer->saveImage(); break;
            case CommandType::SAVE_SCENE: saveScene(app);          break;
            default: break;
        }
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
