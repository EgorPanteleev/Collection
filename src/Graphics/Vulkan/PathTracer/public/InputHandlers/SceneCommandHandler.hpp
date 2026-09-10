//
// Created by igor on 6/12/26.
//

#ifndef COLLECTION_SCENECOMMANDHANDLER_HPP
#define COLLECTION_SCENECOMMANDHANDLER_HPP

#include "Command.hpp"
#include "InputHandlers/CommandContext.hpp"

namespace crv::graphics::vulkan {
    class Model;

    class SceneCommandHandler {
    public:
        void apply(const Command& command, const CommandContext& context) const;
    private:
        void saveScene(Model& model) const;
    };
}

#endif //COLLECTION_SCENECOMMANDHANDLER_HPP
