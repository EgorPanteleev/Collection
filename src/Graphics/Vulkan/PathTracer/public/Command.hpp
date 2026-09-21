//
// Created by igor on 6/12/26.
//

#ifndef COLLECTION_COMMAND_HPP
#define COLLECTION_COMMAND_HPP

#include <cstdint>
#include <string>
#include <variant>
#include <vector>

#include "Model/Material.hpp"
#include "Model/InstanceData.hpp"
#include "Model/DirectLight.hpp"

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

        CLEAR_SELECTION = static_cast<uint32_t>(CommandTarget::SCENE),
        SELECT_INSTANCE,
        DUPLICATE_INSTANCES,
        REMOVE_INSTANCES,
        REMOVE_SELECTED_INSTANCES,
        ADD_MATERIAL,
        ADD_MODEL,
        UPLOAD_TEXTURE,
        LOAD_SKYBOX,
        REMOVE_SKYBOX,
        TRANSFORM_INSTANCES,
        SET_INSTANCE_TRANSFORM,
        SET_INSTANCE_MATERIAL,
        SET_INSTANCE_NAME,
        SET_MATERIAL,
        SET_SKY_COLOR,
        SET_DIRECT_LIGHT,
        SAVE_SCENE,

        PICK_OBJECT = static_cast<uint32_t>(CommandTarget::VIEW),
        REGION_SELECT,
        UPDATE_IMAGE,
        TOGGLE_CONTROL_PANEL,

        QUIT = static_cast<uint32_t>(CommandTarget::APP),
        SAVE_IMAGE,
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
    struct ModelImportPayload    { std::string path; };
    struct SkyboxPayload         { std::string path; };
    struct SetMaterialPayload    { uint32_t index; Material material; };
    struct TransformInstancesPayload   { std::vector<uint32_t> indices; glm::mat4 delta; };
    struct SetInstanceTransformPayload { uint32_t index; Transform transform; };
    struct SetInstanceMaterialPayload  { uint32_t instanceIndex; uint32_t materialIndex; };
    struct SetInstanceNamePayload      { uint32_t index; std::string name; };
    struct SkyColorPayload             { glm::vec3 color; };
    struct DirectLightPayload          { DirectLight light; };

    using CommandPayload = std::variant<
        EmptyPayload,
        SelectInstancePayload,
        RegionSelectPayload,
        InstancesPayload,
        MaterialPayload,
        UploadTexturePayload,
        ModelImportPayload,
        SkyboxPayload,
        SetMaterialPayload,
        TransformInstancesPayload,
        SetInstanceTransformPayload,
        SetInstanceMaterialPayload,
        SetInstanceNamePayload,
        SkyColorPayload,
        DirectLightPayload>;

    struct Command {
        CommandType    type    = CommandType::NONE;
        CommandPayload payload = EmptyPayload{};
    };
}

#endif //COLLECTION_COMMAND_HPP
