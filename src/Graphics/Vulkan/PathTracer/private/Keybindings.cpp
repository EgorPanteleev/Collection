//
// Created by igor on 6/8/26.
//

#include "Keybindings.hpp"

#include <imgui_internal.h>

namespace crv::graphics::vulkan {
    static bool inputBlocked() {
        return ImGui::GetTopMostPopupModal() != nullptr;
    }

    void processInput(const InputState& input, CommandStream& commands) {
        if (inputBlocked()) return;
        if (input.wasPressed(Key::Q)) commands.push(CommandType::QUIT);
        if (input.wasPressed(Key::Z)) commands.push(CommandType::TOGGLE_CONTROL_PANEL);
        if (input.wasPressed(Key::ESCAPE)) commands.push(CommandType::CLEAR_SELECTION);
        if (input.wasPressed(Key::X)) commands.push(CommandType::PICK_OBJECT);
        if (input.wasReleased(MouseButton::MIDDLE)) commands.push(CommandType::PICK_OBJECT);

        if (input.isPressed(Key::W)) commands.push(CommandType::MOVE_FORWARD);
        if (input.isPressed(Key::S)) commands.push(CommandType::MOVE_BACKWARD);
        if (input.isPressed(Key::A)) commands.push(CommandType::MOVE_LEFT);
        if (input.isPressed(Key::D)) commands.push(CommandType::MOVE_RIGHT);
        if (input.isPressed(Key::SPACE)) commands.push(CommandType::MOVE_UP);
        if (input.isPressed(Key::LEFT_CONTROL)) commands.push(CommandType::MOVE_DOWN);

        if (input.isPressed(Key::LEFT))  commands.push(CommandType::ROTATE_LEFT);
        if (input.isPressed(Key::RIGHT)) commands.push(CommandType::ROTATE_RIGHT);
        if (input.isPressed(Key::UP))    commands.push(CommandType::ROTATE_UP);
        if (input.isPressed(Key::DOWN))  commands.push(CommandType::ROTATE_DOWN);

        if (input.isPressed(MouseButton::RIGHT)) {
            const glm::dvec2 delta = input.cursorDelta();
            if (delta.x != 0.0 || delta.y != 0.0) commands.push(CommandType::LOOK);
        }
        if (input.scrollDelta().y != 0.0 && !ImGui::GetIO().WantCaptureMouse)
            commands.push(CommandType::ZOOM);
    }
}
