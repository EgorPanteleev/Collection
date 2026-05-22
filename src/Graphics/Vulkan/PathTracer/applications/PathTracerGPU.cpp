//
// Created by igor on 4/22/26.
//

#include "PathTracerApp.hpp"

namespace cvk = crv::graphics::vulkan;
using Vec4 = cvk::Vec4;

int main() {
    cvk::PathTracerAppCreateInfo createInfo {
        .scenePath = ASSETS_PATH"cornell.json",
        .directLight = cvk::AlignedDirectLight(Vec4(-0.468, 0.318, -0.824, 1), 2.0)
    };
    cvk::PathTracerApp app(createInfo);
    app.run();
}