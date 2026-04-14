//
// Created by igor on 4/7/26.
//

#include "PathTracer.hpp"

namespace cvk = crv::graphics::vulkan;

int main() {
    const cvk::WindowCreateInfo windowCreateInfo{
        .width = 800,
        .height = 600,
        .name = "GPU Path Tracer"
    };
    cvk::PathTracer pathTracer(windowCreateInfo);
    pathTracer.run();
}
