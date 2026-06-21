//
// Created by igor on 6/10/26.
//

#ifndef COLLECTION_MATERIAL_HPP
#define COLLECTION_MATERIAL_HPP

#include <string>
#include <glm/glm.hpp>
#include "SharedTypes.h"

namespace crv::graphics::vulkan {
    struct Material {
        using GPU = MaterialGPU;
        [[nodiscard]] GPU gpu() const;
        [[nodiscard]] static std::vector<GPU> gpu(const std::vector<Material>& materials);

        std::string name{};
        std::string baseColorTexName{};
        glm::vec3   baseColor{};
        glm::vec3   emissionColor     = {1, 1, 1};
        float       luminance         = 0;
        float       metalness         = 0.0f;
        float       roughness         = 1.0f;
        float       ior               = 1.5f;
        float       specular          = 1.0f;
        float       transmission      = 0.0f;
        uint32_t    baseColorTexIndex = UINT32_MAX;
        uint32_t    normalTexIndex    = UINT32_MAX;
    };

    inline Material::GPU Material::gpu() const {
        return {
            .baseColor = baseColor,
            .emissionColor = emissionColor,
            .luminance = luminance,
            .metalness = metalness,
            .roughness = roughness,
            .ior = ior,
            .specular = specular,
            .transmission = transmission,
            .baseColorTexIndex = baseColorTexIndex,
            .normalTexIndex = normalTexIndex,
        };
    }

    inline std::vector<Material::GPU> Material::gpu(const std::vector<Material>& materials) {
        std::vector<GPU> res{};
        res.reserve(materials.size());
        for (const auto& material: materials) res.push_back(material.gpu());
        return res;
    }
}

#endif //COLLECTION_MATERIAL_HPP