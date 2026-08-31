//
// Created by igor on 6/12/26.
//

#ifndef COLLECTION_CAMERAINPUTHANDLER_HPP
#define COLLECTION_CAMERAINPUTHANDLER_HPP

#include "Command.hpp"
#include "AbsCamera.hpp"
#include "InputState.hpp"

namespace crv::graphics::vulkan {
    namespace cs = scene;
    struct CameraInput {
        cs::AbsCamera*    camera    = nullptr;
        const InputState* input     = nullptr;
        float             deltaTime = 0.0f;
    };

    class CameraInputHandler {
    public:
        bool apply(Command command, const CameraInput& context) const;
    };
}

#endif //COLLECTION_CAMERAINPUTHANDLER_HPP
