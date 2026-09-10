//
// Created by igor on 6/8/26.
//

#ifndef COLLECTION_KEYBINDINGS_HPP
#define COLLECTION_KEYBINDINGS_HPP

#include "InputState.hpp"
#include "CommandStream.hpp"

namespace crv::graphics::vulkan {
    void processInput(const InputState& input, CommandStream& commands);
}

#endif //COLLECTION_KEYBINDINGS_HPP
