//
// Created by igor on 6/7/26.
//

#include "PathTracerApp.hpp"

namespace cvk = crv::graphics::vulkan;

int main() {
    const cvk::PathTracerAppCreateInfo createInfo {
        .scenePath = ASSETS_PATH"cornell.json",
        //.directLight = cvk::DirectLightGPU(glm::vec4(-0.468, 0.318, -0.824, 1), 2.0)
    };
    cvk::PathTracerApp app(createInfo);
    //app.run();
}