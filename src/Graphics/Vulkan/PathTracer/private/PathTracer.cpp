//
// Created by igor on 4/8/26.
//

#include "PathTracer.hpp"
#include "Message.hpp"

namespace crv::graphics::vulkan {
    PathTracer::PathTracer(const WindowCreateInfo& windowCreateInfo) {
        bool debug = false;
        #ifndef NDEBUG
            debug = true;
        #endif
        const ContextCreateInfo contextCreateInfo {
            .windowCreateInfo = windowCreateInfo,
            .validationLayers = { "VK_LAYER_KHRONOS_validation" },
            .deviceExtensions = { VK_KHR_SWAPCHAIN_EXTENSION_NAME,
                                     VK_KHR_MAINTENANCE_1_EXTENSION_NAME },
            .enableValidationLayers = debug
        };
        mContext = Context(contextCreateInfo);
    }
}