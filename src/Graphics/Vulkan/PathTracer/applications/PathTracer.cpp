//
// Created by igor on 6/7/26.
//

#include "PathTracerApp.hpp"

#include <string>

namespace cvk = crv::graphics::vulkan;

int main(int argc, char** argv) {
    cvk::PathTracerAppCreateInfo createInfo {
        .scenePath = SCENES_PATH"sponza.json",
    };
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "--scene" && i + 1 < argc) createInfo.scenePath = argv[++i];
    }
    cvk::PathTracerApp app(createInfo);
    app.run();
}
