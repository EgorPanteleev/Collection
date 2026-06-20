//
// Created by igor on 6/10/26.
//

#ifndef COLLECTION_TYPES_HPP
#define COLLECTION_TYPES_HPP

#include "Types/InstanceData.hpp"
#include "Types/Material.hpp"
#include "Types/BLASData.hpp"
#include "Types/UBOData.hpp"
#include "Types/SSBOData.hpp"
#include "SharedTypes.h"

namespace crv::graphics::vulkan {
    using ivec2 = glm::vec<2, uint32_t>;
    struct PushConstants {
        uint32_t frameCount  = 0;
        uint32_t spp         = 1;
        uint32_t minDepth    = 1;
        uint32_t maxDepth    = 1;
        uint32_t displayMode = 4;
        uint32_t nee         = 1;
        uint32_t emissiveCount = 0;
    };

    struct PostprocessPushConstants {
        float    exposure = 1.0f;
        uint32_t tonemap  = 1;
    };

    struct alignas(16) CameraGPU {
        glm::mat4 invView;
        glm::mat4 invProj;
    };

    struct DirectLight {
        glm::vec3 dir{};
        float intensity = 0;
    };

    struct alignas(16) MVPGPU {
        glm::mat4 model, view, proj, trInvModel;
    };
}

#endif //COLLECTION_TYPES_HPP