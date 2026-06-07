//
// Created by igor on 4/22/26.
//

#include "HybridApp.hpp"

namespace cvk = crv::graphics::vulkan;

int main() {
    const cvk::HybridAppCreateInfo createInfo {
        .scenePath = ASSETS_PATH"cornell.json",
        .directLight = cvk::DirectLightGPU(glm::vec4(-0.468, 0.318, -0.824, 1), 2.0)
    };
    cvk::HybridApp app(createInfo);
    app.run();
}