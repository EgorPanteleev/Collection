//
// Created by igor on 6/12/26.
//

#ifndef COLLECTION_COMMAND_HPP
#define COLLECTION_COMMAND_HPP

#include <cstdint>
#include <string>
#include <variant>
#include <vector>

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
        SET_CAMERA_FLY,
        SET_CAMERA_ORBITAL,

        PICK_OBJECT = static_cast<uint32_t>(CommandTarget::SCENE),
        CLEAR_SELECTION,
        SELECT_INSTANCE,
        REGION_SELECT,
        DUPLICATE_INSTANCES,
        REMOVE_INSTANCES,
        ADD_MATERIAL,
        UPLOAD_TEXTURE,
        LOAD_SKYBOX,
        REMOVE_SKYBOX,

        TOGGLE_CONTROL_PANEL = static_cast<uint32_t>(CommandTarget::VIEW),
        UPDATE_IMAGE,

        QUIT = static_cast<uint32_t>(CommandTarget::APP),
        SAVE_IMAGE,
        SAVE_SCENE,
    };

    constexpr CommandTarget commandTarget(const CommandType type) {
        return static_cast<CommandTarget>(static_cast<uint32_t>(type) & COMMAND_TARGET_MASK);
    }

    struct EmptyPayload {};
    struct SelectInstancePayload { uint32_t index; bool additive; };
    struct RegionSelectPayload   { int x0, y0, x1, y1; bool additive; };
    struct InstancesPayload      { std::vector<uint32_t> indices; };
    struct MaterialPayload       { uint32_t instanceIndex; };
    struct UploadTexturePayload  { std::string path; uint32_t materialIndex; int textureType; };
    struct SkyboxPayload         { std::string path; };

    using CommandPayload = std::variant<
        EmptyPayload,
        SelectInstancePayload,
        RegionSelectPayload,
        InstancesPayload,
        MaterialPayload,
        UploadTexturePayload,
        SkyboxPayload>;

    struct Command {
        CommandType    type    = CommandType::NONE;
        CommandPayload payload = EmptyPayload{};
    };
}

#endif //COLLECTION_COMMAND_HPP
