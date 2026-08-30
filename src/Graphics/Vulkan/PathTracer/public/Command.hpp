//
// Created by igor on 6/12/26.
//

#ifndef COLLECTION_COMMAND_HPP
#define COLLECTION_COMMAND_HPP

#include <cstdint>

namespace crv::graphics::vulkan {
    enum class CommandTarget : uint8_t {
        NONE   = 0,
        CAMERA = 1,
        SCENE  = 2,
        VIEW   = 3,
        APP    = 4,
    };

    inline constexpr uint32_t COMMAND_TARGET_SHIFT = 24;

    constexpr uint32_t makeCommand(const CommandTarget target, const uint32_t action) {
        return (static_cast<uint32_t>(target) << COMMAND_TARGET_SHIFT) | action;
    }

    enum class CommandType : uint32_t {
        NONE = 0,
        MOVE_FORWARD  = makeCommand(CommandTarget::CAMERA, 1),
        MOVE_BACKWARD = makeCommand(CommandTarget::CAMERA, 2),
        MOVE_LEFT     = makeCommand(CommandTarget::CAMERA, 3),
        MOVE_RIGHT    = makeCommand(CommandTarget::CAMERA, 4),
        MOVE_UP       = makeCommand(CommandTarget::CAMERA, 5),
        MOVE_DOWN     = makeCommand(CommandTarget::CAMERA, 6),
        ROTATE_LEFT   = makeCommand(CommandTarget::CAMERA, 7),
        ROTATE_RIGHT  = makeCommand(CommandTarget::CAMERA, 8),
        ROTATE_UP     = makeCommand(CommandTarget::CAMERA, 9),
        ROTATE_DOWN   = makeCommand(CommandTarget::CAMERA, 10),
        ZOOM_IN       = makeCommand(CommandTarget::CAMERA, 11),
        ZOOM_OUT      = makeCommand(CommandTarget::CAMERA, 12),

        PICK_OBJECT     = makeCommand(CommandTarget::SCENE, 1),
        CLEAR_SELECTION = makeCommand(CommandTarget::SCENE, 2),

        TOGGLE_CONTROL_PANEL = makeCommand(CommandTarget::VIEW, 1),

        QUIT = makeCommand(CommandTarget::APP, 1),
    };

    constexpr CommandTarget commandTarget(const CommandType type) {
        return static_cast<CommandTarget>(static_cast<uint32_t>(type) >> COMMAND_TARGET_SHIFT);
    }

    struct Command {
        CommandType type = CommandType::NONE;
    };
}

#endif //COLLECTION_COMMAND_HPP
