//
// Created by igor on 6/12/26.
//

#include "Model/Model.hpp"

#include <algorithm>
#include <filesystem>
#include <limits>

namespace crv::graphics::vulkan {
    namespace {
        std::string relativeToAssets(const std::string& path) {
            std::error_code ec;
            const std::filesystem::path rel = std::filesystem::relative(path, ASSETS_PATH, ec);
            return (!ec && !rel.empty()) ? rel.generic_string() : path;
        }
    }

    void Model::load(const json& scene) {
        mScene.load(scene);
        const auto& camera = scene["camera"];
        const auto& window = scene["window"];
        const cs::CameraCreateInfo info {
            .type = camera["type"] == "Fly" ? cs::CameraType::FLY : cs::CameraType::ORBITAL,
            .pos = toVec3(camera["position"]),
            .target = toVec3(camera["target"]),
            .up = toVec3(camera["up"]),
            .zoom = camera["zoom"],
            .FOV = camera["fov"],
            .aspectRatio = static_cast<float>(window["width"]) / static_cast<float>(window["height"]),
            .nearPlane = camera["nearPlane"],
            .farPlane = camera["farPlane"]
        };
        mFlyCamera = cs::FlyCamera(info);
        mOrbitalCamera = cs::OrbitalCamera(info);
        mCamera = info.type == cs::CameraType::FLY ? static_cast<cs::AbsCamera*>(&mFlyCamera)
                                                   : static_cast<cs::AbsCamera*>(&mOrbitalCamera);
    }

    void Model::setActiveCamera(const cs::CameraType type) {
        if (type == cs::CameraType::FLY) {
            mCamera = &mFlyCamera;
            mCamera->setPosition(mOrbitalCamera.position());
            mCamera->setOrientation(mOrbitalCamera.orientation());
        } else {
            mCamera = &mOrbitalCamera;
        }
    }

    uint32_t Model::addMaterial(const uint32_t instanceIndex) {
        auto& instances = mScene.mInstances;
        if (instanceIndex >= instances.size()) return UINT32_MAX;
        Material material = mScene.mMaterials[instances[instanceIndex].materialIndex];
        material.name += " copy";
        mScene.mMaterials.push_back(material);
        const auto index = static_cast<uint32_t>(mScene.mMaterials.size() - 1);
        instances[instanceIndex].materialIndex = index;
        mUpdateState.updateMaterials = true;
        mUpdateState.dirtyInstances.push_back({instanceIndex, InstanceUpdate::Data});
        mUpdateState.resetAccumulation = true;
        return index;
    }

    void Model::duplicateInstances(const std::vector<uint32_t>& indices) {
        auto& instances = mScene.mInstances;
        std::vector<uint32_t> created;
        created.reserve(indices.size());
        for (const uint32_t index : indices) {
            if (index >= instances.size()) continue;
            instances.push_back(instances[index]);
            created.push_back(static_cast<uint32_t>(instances.size() - 1));
        }
        if (created.empty()) return;
        mSelection.selectedInstances = created;
        mSelection.activeInstance = created.back();
        mSelection.pending = false;
        mUpdateState.updateInstances = true;
        mUpdateState.resetAccumulation = true;
    }

    void Model::removeInstances(const std::vector<uint32_t>& indices) {
        auto& instances = mScene.mInstances;
        std::vector<uint32_t> sorted(indices);
        std::sort(sorted.begin(), sorted.end(), std::greater<>());
        sorted.erase(std::unique(sorted.begin(), sorted.end()), sorted.end());
        if (sorted.empty() || sorted.size() >= instances.size()) return;
        for (const uint32_t index : sorted) {
            if (index < instances.size()) instances.erase(instances.begin() + index);
        }
        clearSelection();
        mUpdateState.updateInstances = true;
        mUpdateState.resetAccumulation = true;
    }

    void Model::addTexture(const std::string& path, const uint32_t materialIndex, const int textureType) {
        if (materialIndex >= mScene.mMaterials.size()) return;
        cm::Texture::Type type;
        switch (textureType) {
            case 1:  type = cm::Texture::NORMAL;             break;
            case 2:  type = cm::Texture::METAL_ROUGHNESS;    break;
            case 3:  type = cm::Texture::CLEARCOAT;          break;
            case 4:  type = cm::Texture::CLEARCOAT_ROUGHNESS; break;
            default: type = cm::Texture::BASE_COLOR;         break;
        }
        mScene.mTextureSources.push_back(cm::AbsLoader::loadTexture(path, type));
        const auto index = static_cast<uint32_t>(mScene.mTextureSources.size() - 1);
        const std::string name = std::filesystem::path(path).filename().string();
        const std::string rel  = relativeToAssets(path);
        Material& material = mScene.mMaterials[materialIndex];
        switch (textureType) {
            case 1:  material.normalTexIndex = index; material.normalTexName = name; material.normalTexPath = rel; break;
            case 2:  material.metalRoughnessTexIndex = index; material.metalRoughnessTexName = name; material.metalRoughnessTexPath = rel; break;
            case 3:  material.clearcoatTexIndex = index; material.clearcoatTexName = name; material.clearcoatTexPath = rel; break;
            case 4:  material.clearcoatRoughnessTexIndex = index; material.clearcoatRoughnessTexName = name; material.clearcoatRoughnessTexPath = rel; break;
            default: material.baseColorTexIndex = index; material.baseColorTexName = name; material.baseColorTexPath = rel; break;
        }
        mUpdateState.dirtyTextures.push_back(index);
        mUpdateState.dirtyMaterials.push_back(materialIndex);
    }

    void Model::loadSkybox(const std::string& path) {
        mScene.mTextureSources.push_back(cm::AbsLoader::loadSkybox(path));
        mScene.mSkyboxIndex = static_cast<uint32_t>(mScene.mTextureSources.size() - 1);
        mScene.mSkyboxName = std::filesystem::path(path).filename().string();
        mScene.mSkyboxPath = relativeToAssets(path);
        mUpdateState.updateSkybox = true;
        mUpdateState.resetAccumulation = true;
    }

    void Model::removeSkybox() {
        mScene.mSkyboxIndex = UINT32_MAX;
        mScene.mSkyboxName.clear();
        mScene.mSkyboxPath.clear();
        mUpdateState.updateSkybox = true;
        mUpdateState.resetAccumulation = true;
    }

    void Model::clearSelection() {
        mSelection.selectedInstances.clear();
        mSelection.activeInstance = UINT32_MAX;
        mSelection.pending = false;
    }

    void Model::select(const uint32_t id, const bool additive) {
        if (id == 0) {
            if (!additive) clearSelection();
            return;
        }
        const uint32_t index = id - 1;
        auto& selected = mSelection.selectedInstances;
        const auto it = std::find(selected.begin(), selected.end(), index);
        if (!additive) {
            selected = {index};
            mSelection.activeInstance = index;
        } else if (it != selected.end()) {
            selected.erase(it);
            mSelection.activeInstance = selected.empty() ? UINT32_MAX : selected.back();
        } else {
            selected.push_back(index);
            mSelection.activeInstance = index;
        }
    }

    void Model::regionSelect(int x0, int y0, int x1, int y1, bool additive, uint32_t width, uint32_t height) {
        const float fw = static_cast<float>(width);
        const float fh = static_cast<float>(height);
        const float nx0 = static_cast<float>(std::min(x0, x1)) / fw * 2.0f - 1.0f;
        const float nx1 = static_cast<float>(std::max(x0, x1)) / fw * 2.0f - 1.0f;
        const float ny0 = static_cast<float>(std::min(y0, y1)) / fh * 2.0f - 1.0f;
        const float ny1 = static_cast<float>(std::max(y0, y1)) / fh * 2.0f - 1.0f;

        const glm::mat4 viewProj = mCamera->projectionMatrix() * mCamera->viewMatrix();
        const auto& instances = mScene.mInstances;
        const auto& meshes = mScene.mMeshes;
        auto& selected = mSelection.selectedInstances;

        if (!additive) selected.clear();
        for (uint32_t i = 0; i < instances.size(); ++i) {
            const InstanceData& instance = instances[i];
            const MeshData& mesh = meshes[instance.meshIndex];
            const glm::mat4 mvp = viewProj * instance.transform.matrix();

            glm::vec2 boxMin(std::numeric_limits<float>::max());
            glm::vec2 boxMax(std::numeric_limits<float>::lowest());
            bool anyInFront = false;
            for (int c = 0; c < 8; ++c) {
                const glm::vec4 corner {
                    (c & 1) ? mesh.bbox.max.x : mesh.bbox.min.x,
                    (c & 2) ? mesh.bbox.max.y : mesh.bbox.min.y,
                    (c & 4) ? mesh.bbox.max.z : mesh.bbox.min.z,
                    1.0f
                };
                const glm::vec4 clip = mvp * corner;
                if (clip.w <= 1e-4f) continue;
                anyInFront = true;
                const glm::vec2 ndc = glm::vec2(clip) / clip.w;
                boxMin = glm::min(boxMin, ndc);
                boxMax = glm::max(boxMax, ndc);
            }
            if (!anyInFront) continue;
            if (boxMax.x < nx0 || boxMin.x > nx1 || boxMax.y < ny0 || boxMin.y > ny1) continue;

            if (std::find(selected.begin(), selected.end(), i) == selected.end())
                selected.push_back(i);
        }
        mSelection.activeInstance = selected.empty() ? UINT32_MAX : selected.back();
    }
}
