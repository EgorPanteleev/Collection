//
// Created by igor on 6/12/26.
//

#include "InputHandlers/SceneCommandHandler.hpp"
#include "Model/Model.hpp"

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

    void SceneCommandHandler::apply(const Command& command, const CommandContext& context) const {
        Model&       model = *context.model;
        UpdateState& state = model.updateState();
        switch (command.type) {
            case CommandType::CLEAR_SELECTION: model.clearSelection(); break;
            case CommandType::SELECT_INSTANCE: {
                const auto& p = std::get<SelectInstancePayload>(command.payload);
                model.selectInstance(p.index, p.additive);
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
                    state.markInstanceDirty(index);
                break;
            case CommandType::UPDATE_INSTANCE_DATA:
                state.markInstanceDataDirty(std::get<IndexPayload>(command.payload).index);
                break;
            case CommandType::UPDATE_MATERIAL:
                state.markMaterialDirty(std::get<IndexPayload>(command.payload).index);
                break;

            case CommandType::SAVE_SCENE: saveScene(model); break;
            default: break;
        }
    }

    void SceneCommandHandler::saveScene(Model& model) const {
        const json scene = model.scene().save();
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
