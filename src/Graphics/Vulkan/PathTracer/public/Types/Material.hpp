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
        std::string normalTexName{};
        std::string metalRoughnessTexName{};
        std::string clearcoatTexName{};
        std::string clearcoatRoughnessTexName{};
        std::string baseColorTexPath{};
        std::string normalTexPath{};
        std::string metalRoughnessTexPath{};
        std::string clearcoatTexPath{};
        std::string clearcoatRoughnessTexPath{};
        glm::vec3   baseColor{};
        float       luminance         = 0;
        float       metalness         = 0.0f;
        float       roughness         = 0.0f;
        float       ior               = 1.5f;
        float       specular          = 0.0f;
        float       transmission      = 0.0f;
        float       clearcoat          = 0.0f;
        float       clearcoatRoughness = 0.0f;
        glm::vec3   absorption         = {1, 1, 1};
        float       opacity            = 1.0f;
        float       normalScale        = 1.0f;
        float       anisotropy         = 0.0f;
        float       sheen              = 0.0f;
        uint32_t    baseColorTexIndex = UINT32_MAX;
        uint32_t    normalTexIndex    = UINT32_MAX;
        uint32_t    metalRoughnessTexIndex = UINT32_MAX;
        uint32_t    clearcoatTexIndex = UINT32_MAX;
        uint32_t    clearcoatRoughnessTexIndex = UINT32_MAX;
    };

    inline Material::GPU Material::gpu() const {
        return {
            .baseColor = baseColor,
            .luminance = luminance,
            .metalness = metalness,
            .roughness = roughness,
            .ior = ior,
            .specular = specular,
            .transmission = transmission,
            .clearcoat = clearcoat,
            .clearcoatRoughness = clearcoatRoughness,
            .absorption = absorption,
            .baseColorTexIndex = baseColorTexIndex,
            .normalTexIndex = normalTexIndex,
            .metalRoughnessTexIndex = metalRoughnessTexIndex,
            .clearcoatTexIndex = clearcoatTexIndex,
            .clearcoatRoughnessTexIndex = clearcoatRoughnessTexIndex,
            .opacity = opacity,
            .normalScale = normalScale,
            .anisotropy = anisotropy,
            .sheen = sheen,
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