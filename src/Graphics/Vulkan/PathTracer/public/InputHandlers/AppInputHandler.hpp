//
// Created by igor on 6/12/26.
//

#ifndef COLLECTION_APPINPUTHANDLER_HPP
#define COLLECTION_APPINPUTHANDLER_HPP

#include "Command.hpp"

namespace crv::graphics::vulkan {
    class PathTracerApp;

    class AppInputHandler {
    public:
        void apply(const Command& command, PathTracerApp* app) const;
    };
}

#endif //COLLECTION_APPINPUTHANDLER_HPP
