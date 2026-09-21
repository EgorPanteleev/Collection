//
// Created by igor on 6/10/26.
//

#ifndef COLLECTION_VIEW_TYPES_HPP
#define COLLECTION_VIEW_TYPES_HPP

#include "View/BLASData.hpp"
#include "View/UBOBuilder.hpp"
#include "View/SSBOBuilder.hpp"
#include "SharedTypes.h"

#include <glm/glm.hpp>
#include <cstdint>

namespace crv::graphics::vulkan {
    using ivec2 = glm::vec<2, uint32_t>;

    struct PushConstants {
        uint32_t frameCount  = 0;
        uint32_t spp         = 1;
        uint32_t minDepth    = 1;
        uint32_t maxDepth    = 1;
        uint32_t displayMode = 0;
        uint32_t nee         = 1;
        uint32_t emissiveCount = 0;
        uint32_t skyboxIndex = UINT32_MAX;
        float    envIntegral = 0.0f;
        uint32_t envNee      = 1;
        float    aperture      = 0.0f;
        float    focusDistance = 10.0f;
        uint64_t envMarginalCdfAddr = 0;
        uint64_t envCondCdfAddr     = 0;
        uint64_t envCondFuncAddr    = 0;
        glm::vec3 skyColor = glm::vec3(0.1f);
        float    emissivePowerInv = 0.0f;
    };

    struct PostprocessPushConstants {
        float    exposure             = 1.0f;
        uint32_t tonemapMode          = 1;
        uint32_t displayMode          = 0;
        uint32_t renderScale          = 1;
        uint32_t autoExposure         = 0;
        float    deltaTime            = 0.0f;
        uint32_t renderWidth          = 1;
        uint32_t renderHeight         = 1;
        float    minLogLuminance      = -10.0f;
        float    logLuminanceRange    = 20.0f;
    };

    struct alignas(16) CameraGPU {
        glm::mat4 invView;
        glm::mat4 invProj;
    };

    struct alignas(16) MVPGPU {
        glm::mat4 model, view, proj, trInvModel;
    };
}

#endif //COLLECTION_VIEW_TYPES_HPP
