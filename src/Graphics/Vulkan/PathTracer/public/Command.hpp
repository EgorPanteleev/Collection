//
// Created by igor on 6/12/26.
//

#ifndef COLLECTION_COMMAND_HPP
#define COLLECTION_COMMAND_HPP

#include <cstdint>
#include <variant>

namespace crv::graphics::vulkan {
    inline constexpr uint32_t COMMAND_TARGET_SHIFT = 24;
    inline constexpr uint32_t COMMAND_TARGET_MASK  = 0xFFu << COMMAND_TARGET_SHIFT;

    enum class CommandTarget : uint32_t {
        CAMERA = 1u << COMMAND_TARGET_SHIFT,
        SCENE  = 2u << COMMAND_TARGET_SHIFT,
        VIEW   = 3u << COMMAND_TARGET_SHIFT,
        APP    = 4u << COMMAND_TARGET_SHIFT,
    };

    enum class CommandType : uint32_t {
        NONE = 0,
        MOVE_FORWARD = static_cast<uint32_t>(CommandTarget::CAMERA),
        MOVE_BACKWARD,
        MOVE_LEFT,
        MOVE_RIGHT,
        MOVE_UP,
        MOVE_DOWN,
        ROTATE_LEFT,
        ROTATE_RIGHT,
        ROTATE_UP,
        ROTATE_DOWN,
        LOOK,
        ZOOM,

        PICK_OBJECT = static_cast<uint32_t>(CommandTarget::SCENE),
        CLEAR_SELECTION,

        TOGGLE_CONTROL_PANEL = static_cast<uint32_t>(CommandTarget::VIEW),

        QUIT = static_cast<uint32_t>(CommandTarget::APP),
    };

    constexpr CommandTarget commandTarget(const CommandType type) {
        return static_cast<CommandTarget>(static_cast<uint32_t>(type) & COMMAND_TARGET_MASK);
    }

    struct EmptyPayload {};

    using CommandPayload = std::variant<EmptyPayload>;

    struct Command {
        CommandType    type    = CommandType::NONE;
        CommandPayload payload = EmptyPayload{};
    };
}

#endif //COLLECTION_COMMAND_HPP
