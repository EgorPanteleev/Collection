//
// Created by igor on 6/7/26.
//

#include "PathTracerApp.hpp"

namespace cvk = crv::graphics::vulkan;

int main() {
    const cvk::PathTracerAppCreateInfo createInfo {
        .scenePath = ASSETS_PATH"sponza.json",
    };
    cvk::PathTracerApp app(createInfo);
    app.run();
}
