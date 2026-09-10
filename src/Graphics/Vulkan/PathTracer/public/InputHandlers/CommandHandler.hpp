//
// Created by igor on 6/12/26.
//

#ifndef COLLECTION_COMMANDHANDLER_HPP
#define COLLECTION_COMMANDHANDLER_HPP

#include "CommandStream.hpp"
#include "InputHandlers/CommandContext.hpp"
#include "InputHandlers/CameraCommandHandler.hpp"
#include "InputHandlers/SceneCommandHandler.hpp"
#include "InputHandlers/RendererCommandHandler.hpp"

namespace crv::graphics {
    class InputState;
}

namespace crv::graphics::vulkan {
    class Model;
    class Renderer;

    struct CommandHandlerCreateInfo {
        Model*            model    = nullptr;
        Renderer*         renderer = nullptr;
        const InputState* input    = nullptr;
    };

    class CommandHandler {
    public:
        CommandHandler() = default;
        explicit CommandHandler(const CommandHandlerCreateInfo& info);
        void apply(const CommandStream& commands, float deltaTime) const;
    private:
        Model*                 mModel    = nullptr;
        Renderer*              mRenderer = nullptr;
        const InputState*      mInput    = nullptr;
        CameraCommandHandler   mCameraHandler{};
        SceneCommandHandler    mSceneHandler{};
        RendererCommandHandler mRendererHandler{};
    };
}

#endif //COLLECTION_COMMANDHANDLER_HPP
