//
// Created by igor on 10/11/26.
//

#include "Model/Volume.hpp"

#include <algorithm>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/quaternion.hpp>

namespace crv::graphics::vulkan {
    glm::mat4 Volume::matrix() const {
        const glm::mat4 rotate = glm::toMat4(glm::quat(glm::radians(rotation)));
        return glm::translate(glm::mat4(1.0f), position) * rotate *
               glm::scale(glm::mat4(1.0f), glm::max(scale, glm::vec3(1e-4f)));
    }

    Volume::GPU Volume::gpu() const {
        return {
            .invModel      = glm::inverse(matrix()),
            .color         = color,
            .density       = std::max(density, 0.0f),
            .anisotropy    = std::clamp(anisotropy, -0.95f, 0.95f),
            .noiseScale    = std::max(noiseScale, 0.0f),
            .noiseStrength = std::clamp(noiseStrength, 0.0f, 1.0f),
            .coverage      = std::clamp(coverage, 0.0f, 0.95f),
            .octaves       = static_cast<uint32_t>(std::clamp(octaves, 1, 8)),
            .shape         = static_cast<uint32_t>(shape == BOX ? BOX : ELLIPSOID)
        };
    }

    std::vector<Volume::GPU> Volume::gpu(const std::vector<Volume>& volumes) {
        std::vector<GPU> result;
        result.reserve(volumes.size());
        for (const auto& volume : volumes) result.push_back(volume.gpu());
        return result;
    }
}
