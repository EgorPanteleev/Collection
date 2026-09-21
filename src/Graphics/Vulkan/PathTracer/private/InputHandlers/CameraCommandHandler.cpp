//
// Created by igor on 6/12/26.
//

#include "InputHandlers/CameraCommandHandler.hpp"
#include "Model/Model.hpp"
#include "InputState.hpp"

namespace crv::graphics::vulkan {
    bool CameraCommandHandler::apply(const Command command, const CommandContext& context) const {
        switch (command.type) {
            case CommandType::SET_CAMERA_FLY:     context.model->setActiveCamera(cs::CameraType::FLY);     return false;
            case CommandType::SET_CAMERA_ORBITAL: context.model->setActiveCamera(cs::CameraType::ORBITAL); return false;
            default: break;
        }

        auto* camera = context.model->camera();
        if (!camera) return false;

        const float speed       = 0.06f * context.deltaTime;
        const float rotateSpeed = speed * 0.3f;
        constexpr double lookSensitivity = 0.1;
        constexpr double zoomSpeed       = 10.0;

        switch (command.type) {
            case CommandType::MOVE_FORWARD:  camera->move(speed, 0, 0);   return true;
            case CommandType::MOVE_BACKWARD: camera->move(-speed, 0, 0);  return true;
            case CommandType::MOVE_LEFT:     camera->move(0, -speed, 0);  return true;
            case CommandType::MOVE_RIGHT:    camera->move(0, speed, 0);   return true;
            case CommandType::MOVE_UP:       camera->move(0, 0, speed);   return true;
            case CommandType::MOVE_DOWN:     camera->move(0, 0, -speed);  return true;
            case CommandType::ROTATE_LEFT:   camera->rotate(0, rotateSpeed, 0);  return true;
            case CommandType::ROTATE_RIGHT:  camera->rotate(0, -rotateSpeed, 0); return true;
            case CommandType::ROTATE_UP:     camera->rotate(rotateSpeed, 0, 0);  return true;
            case CommandType::ROTATE_DOWN:   camera->rotate(-rotateSpeed, 0, 0); return true;
            case CommandType::LOOK: {
                const glm::dvec2 lookDelta = context.input->cursorDelta();
                camera->rotate(static_cast<float>(-lookDelta.y * lookSensitivity),
                               static_cast<float>(-lookDelta.x * lookSensitivity), 0.f);
                return true;
            }
            case CommandType::ZOOM:
                camera->zoom(static_cast<float>(context.input->scrollDelta().y * zoomSpeed));
                return true;
            default:
                return false;
        }
    }
}
