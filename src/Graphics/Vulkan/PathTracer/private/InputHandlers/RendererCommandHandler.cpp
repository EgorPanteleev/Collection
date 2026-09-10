//
// Created by igor on 6/12/26.
//

#include "InputHandlers/RendererCommandHandler.hpp"
#include "Model/Model.hpp"
#include "View/Renderer.hpp"
#include "InputState.hpp"

namespace crv::graphics::vulkan {
    void RendererCommandHandler::apply(const Command& command, const CommandContext& context) const {
        Model&            model    = *context.model;
        Renderer&         renderer = *context.renderer;
        const InputState& input    = *context.input;
        switch (command.type) {
            case CommandType::PICK_OBJECT: {
                const bool additive = input.isPressed(Key::LEFT_SHIFT) || input.isPressed(Key::RIGHT_SHIFT);
                renderer.pick(input.cursorPos(), additive);
                break;
            }
            case CommandType::REGION_SELECT: {
                const auto& p = std::get<RegionSelectPayload>(command.payload);
                const auto [width, height] = renderer.extent();
                model.regionSelect(p.x0, p.y0, p.x1, p.y1, p.additive, width, height);
                break;
            }
            case CommandType::UPDATE_IMAGE:         renderer.resetAccumulation(); break;
            case CommandType::TOGGLE_CONTROL_PANEL: renderer.toggleUI();          break;
            case CommandType::SAVE_IMAGE:           renderer.saveImage();         break;
            case CommandType::QUIT:                 renderer.window().close();    break;
            default: break;
        }
    }
}
