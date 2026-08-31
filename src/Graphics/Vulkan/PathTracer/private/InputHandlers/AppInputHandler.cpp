//
// Created by igor on 6/12/26.
//

#include "InputHandlers/AppInputHandler.hpp"
#include "PathTracerApp.hpp"

namespace crv::graphics::vulkan {
    void AppInputHandler::apply(const Command& command, PathTracerApp* app) const {
        switch (command.type) {
            case CommandType::SET_CAMERA_FLY:     app->setCamera(scene::CameraType::FLY);     break;
            case CommandType::SET_CAMERA_ORBITAL: app->setCamera(scene::CameraType::ORBITAL); break;

            case CommandType::PICK_OBJECT:     app->pickAtCursor();  break;
            case CommandType::CLEAR_SELECTION: app->clearSelection(); break;
            case CommandType::SELECT_INSTANCE: {
                const auto& p = std::get<SelectInstancePayload>(command.payload);
                app->selectInstance(p.index, p.additive);
                break;
            }
            case CommandType::REGION_SELECT: {
                const auto& p = std::get<RegionSelectPayload>(command.payload);
                app->regionSelect(p.x0, p.y0, p.x1, p.y1, p.additive);
                break;
            }
            case CommandType::DUPLICATE_INSTANCES:
                app->duplicateInstances(std::get<InstancesPayload>(command.payload).indices);
                break;
            case CommandType::REMOVE_INSTANCES:
                app->removeInstances(std::get<InstancesPayload>(command.payload).indices);
                break;
            case CommandType::ADD_MATERIAL:
                app->addMaterial(std::get<MaterialPayload>(command.payload).instanceIndex);
                break;
            case CommandType::UPLOAD_TEXTURE: {
                const auto& p = std::get<UploadTexturePayload>(command.payload);
                app->uploadTexture(p.path, p.materialIndex, p.textureType);
                break;
            }
            case CommandType::LOAD_SKYBOX:
                app->loadSkybox(std::get<SkyboxPayload>(command.payload).path);
                break;
            case CommandType::REMOVE_SKYBOX: app->removeSkybox(); break;

            case CommandType::UPDATE_IMAGE:         app->updateImage();        break;
            case CommandType::TOGGLE_CONTROL_PANEL: app->toggleControlPanel(); break;

            case CommandType::QUIT:       app->window().close(); break;
            case CommandType::SAVE_IMAGE: app->saveImage();      break;
            case CommandType::SAVE_SCENE: app->saveScene();      break;
            default: break;
        }
    }
}
