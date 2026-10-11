//
// Created by igor on 10/11/26.
//

#ifndef COLLECTION_VOLUME_HPP
#define COLLECTION_VOLUME_HPP

#include <string>
#include <vector>

#include <glm/glm.hpp>

#include "SharedTypes.h"

namespace crv::graphics::vulkan {
    struct Volume {
        enum Shape : uint32_t { ELLIPSOID = 0, BOX = 1 };
        using GPU = VolumeGPU;

        [[nodiscard]] glm::mat4 matrix() const;
        [[nodiscard]] GPU gpu() const;
        [[nodiscard]] static std::vector<GPU> gpu(const std::vector<Volume>& volumes);

        std::string name{"Cloud"};
        glm::vec3   position{0.0f};
        glm::vec3   rotation{0.0f};
        glm::vec3   scale{1.0f};
        glm::vec3   color{1.0f};
        float       density       = 2.0f;
        float       anisotropy    = 0.3f;
        float       noiseScale    = 1.5f;
        float       noiseStrength = 0.75f;
        float       coverage      = 0.35f;
        int         octaves       = 4;
        int         shape         = ELLIPSOID;
    };
}

#endif //COLLECTION_VOLUME_HPP
