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
    private:
        void pick(PathTracerApp* app) const;
        void toggleControlPanel(PathTracerApp* app) const;
        void saveImage(PathTracerApp* app) const;
        void saveScene(PathTracerApp* app) const;
    };
}

#endif //COLLECTION_APPINPUTHANDLER_HPP
