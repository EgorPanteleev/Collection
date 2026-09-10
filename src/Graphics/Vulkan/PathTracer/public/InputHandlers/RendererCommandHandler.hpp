//
// Created by igor on 6/12/26.
//

#ifndef COLLECTION_RENDERERCOMMANDHANDLER_HPP
#define COLLECTION_RENDERERCOMMANDHANDLER_HPP

#include "Command.hpp"
#include "InputHandlers/CommandContext.hpp"

namespace crv::graphics::vulkan {
    class RendererCommandHandler {
    public:
        void apply(const Command& command, const CommandContext& context) const;
    };
}

#endif //COLLECTION_RENDERERCOMMANDHANDLER_HPP
