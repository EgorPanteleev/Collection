//
// Created by igor on 6/12/26.
//

#include "InputHandlers/CommandHandler.hpp"
#include "Model/Model.hpp"

namespace crv::graphics::vulkan {
    CommandHandler::CommandHandler(const CommandHandlerCreateInfo& info)
        : mModel(info.model), mRenderer(info.renderer), mInput(info.input) {}

    void CommandHandler::apply(const CommandStream& commands, const float deltaTime) const {
        const CommandContext context {
            .model     = mModel,
            .renderer  = mRenderer,
            .input     = mInput,
            .deltaTime = deltaTime
        };
        bool cameraMoved = false;
        for (const Command& command : commands.get()) {
            switch (commandTarget(command.type)) {
                case CommandTarget::CAMERA: if (mCameraHandler.apply(command, context)) cameraMoved = true; break;
                case CommandTarget::SCENE:  mSceneHandler.apply(command, context);    break;
                case CommandTarget::VIEW:
                case CommandTarget::APP:    mRendererHandler.apply(command, context); break;
                default: break;
            }
        }
        if (cameraMoved) mModel->updateState().markCameraMoved();
    }
}
