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
}

/* Plan
 * 1) Creating context
 *   1.1) window, surface
 *   1.2) instance
 *   1.3) devices
 *   1.4) queues
 *   1.5) allocator
 *
 * 2) Buffers - resources
 *   2.1) BVH
 *   2.2) Materials
 *   2.3) Lights
 *
 * 3) Bind resources
 *   3.1) Descriptor Set Layout
 *   3.2) Descriptor Set
 *
 * 4) Bind descriptor sets
 *   4.1) Write shaders, compile it
 *   4.2) Compute Pipeline Layout
 *   4.3) Compute Pipeline
 *
 * 5) Command buffers, recording
 *   5.1) Record command buffer
 *   5.2) Submit command buffer
 *
 * 6) Present the results
*/
