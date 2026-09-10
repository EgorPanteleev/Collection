//
// Created by igor on 6/12/26.
//

#ifndef COLLECTION_COMMANDCONTEXT_HPP
#define COLLECTION_COMMANDCONTEXT_HPP

namespace crv::graphics { class InputState; }

namespace crv::graphics::vulkan {
    class Model;
    class Renderer;

    struct CommandContext {
        Model*            model     = nullptr;
        Renderer*         renderer  = nullptr;
        const InputState* input     = nullptr;
        float             deltaTime = 0.0f;
    };
}

#endif //COLLECTION_COMMANDCONTEXT_HPP
