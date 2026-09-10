//
// Created by igor on 6/12/26.
//

#ifndef COLLECTION_CAMERACOMMANDHANDLER_HPP
#define COLLECTION_CAMERACOMMANDHANDLER_HPP

#include "Command.hpp"
#include "InputHandlers/CommandContext.hpp"

namespace crv::graphics::vulkan {
    class CameraCommandHandler {
    public:
        bool apply(Command command, const CommandContext& context) const;
    };
}

#endif //COLLECTION_CAMERACOMMANDHANDLER_HPP
