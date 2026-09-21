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
        const auto& instances = mScene.instances();
        if (instanceIndex >= instances.size()) return UINT32_MAX;
        Material material = mScene.materials()[instances[instanceIndex].materialIndex];
        material.name += " copy";
        const uint32_t index = mScene.addMaterial(material);
        mScene.setInstanceMaterial(instanceIndex, index);
        mUpdateState.updateMaterials = true;
        mUpdateState.markInstanceDataDirty(instanceIndex);
        return index;
    }

    void Model::setMaterial(const uint32_t index, const Material& material) {
        if (index >= mScene.materials().size()) return;
        mScene.setMaterial(index, material);
        mUpdateState.markMaterialDirty(index);
    }

    std::vector<uint32_t> Model::transformRoots(const std::vector<uint32_t>& indices) const {
        const std::vector<bool> affected = mScene.subtreeMask(indices);
        const auto& instances = mScene.instances();
        const auto count = static_cast<uint32_t>(instances.size());
        std::vector<uint32_t> roots;
        roots.reserve(indices.size());
        for (const uint32_t index : indices) {
            if (index >= count) continue;
            const int32_t parent = instances[index].parentIndex;
            if (parent >= 0 && affected[parent]) continue;
            if (std::find(roots.begin(), roots.end(), index) == roots.end()) roots.push_back(index);
        }
        return roots;
    }

    void Model::transformInstances(const std::vector<uint32_t>& indices, const glm::mat4& delta) {
        const auto& instances = mScene.instances();
        const std::vector<uint32_t> roots = transformRoots(indices);
        std::vector<uint32_t> targets;
        std::vector<Transform> locals;
        targets.reserve(roots.size());
        locals.reserve(roots.size());
        for (const uint32_t index : roots) {
            const InstanceData& instance = instances[index];
            const glm::mat4 parentWorld = instance.parentIndex >= 0
                ? instances[instance.parentIndex].world : glm::mat4(1.0f);
            const glm::mat4 newLocal = glm::inverse(parentWorld) * delta * instance.world;
            targets.push_back(index);
            locals.push_back(decomposeTransform(newLocal));
        }
        for (size_t i = 0; i < targets.size(); ++i) mScene.setInstanceTransform(targets[i], locals[i]);
        for (const uint32_t index : targets) markInstanceSubtreeDirty(index);
    }

    void Model::setInstanceTransform(const uint32_t index, const Transform& transform) {
        if (index >= mScene.instances().size()) return;
        mScene.setInstanceTransform(index, transform);
        markInstanceSubtreeDirty(index);
    }

    void Model::markInstanceSubtreeDirty(const uint32_t root) {
        const std::vector<bool> affected = mScene.subtreeMask({root});
        const auto count = static_cast<uint32_t>(affected.size());
        for (uint32_t i = 0; i < count; ++i)
            if (affected[i]) mUpdateState.markInstanceDirty(i);
    }

    void Model::setInstanceMaterial(const uint32_t instanceIndex, const uint32_t materialIndex) {
        if (instanceIndex >= mScene.instances().size() || materialIndex >= mScene.materials().size()) return;
        mScene.setInstanceMaterial(instanceIndex, materialIndex);
        mUpdateState.markInstanceDataDirty(instanceIndex);
    }

    void Model::setSkyColor(const glm::vec3& color) {
        mScene.setSkyColor(color);
        mUpdateState.markImageDirty();
    }

    void Model::setEnvRotation(const float radians) {
        mScene.setEnvRotation(radians);
        mUpdateState.markImageDirty();
    }

    void Model::setDirectLight(const DirectLight& light) {
        mScene.setDirectLight(light);
        mUpdateState.markImageDirty();
    }

    void Model::duplicateInstances(const std::vector<uint32_t>& indices) {
        const std::vector<uint32_t> created = mScene.duplicateInstances(indices);
        if (created.empty()) return;
        mSelection.selectedInstances = created;
        mSelection.activeInstance = created.back();
        mUpdateState.updateInstances = true;
    }

    void Model::removeInstances(const std::vector<uint32_t>& indices) {
        if (indices.empty()) return;
        mScene.removeInstances(indices);
        clearSelection();
        mUpdateState.updateInstances = true;
    }

    void Model::addTexture(const std::string& path, const uint32_t materialIndex, const int textureType) {
        if (materialIndex >= mScene.materials().size()) return;
        cm::Texture::Type type;
        switch (textureType) {
            case 1:  type = cm::Texture::NORMAL;             break;
            case 2:  type = cm::Texture::METAL_ROUGHNESS;    break;
            case 3:  type = cm::Texture::CLEARCOAT;          break;
            case 4:  type = cm::Texture::CLEARCOAT_ROUGHNESS; break;
            default: type = cm::Texture::BASE_COLOR;         break;
        }
        const uint32_t index = mScene.addTextureSource(cm::AbsLoader::loadTexture(path, type));
        mScene.setMaterialTexture(materialIndex, textureType, index, relativeToAssets(path));
        mUpdateState.dirtyTextures.push_back(index);
        mUpdateState.dirtyMaterials.push_back(materialIndex);
    }

    void Model::addModel(const std::string& path) {
        mScene.addModel(relativeToAssets(path));
        clearSelection();
        mUpdateState.addModel = true;
    }

    void Model::loadSkybox(const std::string& path) {
        const uint32_t index = mScene.addTextureSource(cm::AbsLoader::loadSkybox(path));
        mScene.setSkybox(index, relativeToAssets(path));
        mUpdateState.updateSkybox = true;
    }

    void Model::removeSkybox() {
        mScene.clearSkybox();
        mUpdateState.updateSkybox = true;
    }

    void Model::clearSelection() {
        mSelection.selectedInstances.clear();
        mSelection.activeInstance = UINT32_MAX;
    }

    void Model::select(const uint32_t id, const bool additive) {
        if (id == 0 or id - 1 >= mScene.instances().size()) {
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
        const auto& instances = mScene.instances();
        const auto& meshes = mScene.meshes();
        auto& selected = mSelection.selectedInstances;

        if (!additive) selected.clear();
        for (uint32_t i = 0; i < instances.size(); ++i) {
            const InstanceData& instance = instances[i];
            if (instance.isGroup()) continue;
            const MeshData& mesh = meshes[instance.meshIndex];
            const glm::mat4 mvp = viewProj * instance.world;

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
