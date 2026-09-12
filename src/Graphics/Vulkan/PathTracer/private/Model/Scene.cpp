//
// Created by igor on 6/10/26.
//

#include "Model/Scene.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

namespace crv::graphics::vulkan {
    Scene::Scene(const SceneCreateInfo&) {}

    void Scene::load(const json& scene) {
        mJson = scene;
        mExplicit = mJson.value("version", 1) >= 2;
        auto directLight = mJson["directLight"];
        mDirectLight.dir = glm::vec4(toVec3(directLight["direction"]), 1);
        mDirectLight.intensity = directLight["intensity"];
        if (mJson.contains("skyColor") && mJson["skyColor"].is_array()) mSkyColor = toVec3(mJson["skyColor"]);

        loadMaterials();
        std::vector<std::string> models = mJson["modelImports"];
        for (int modelIndex = 0; modelIndex < models.size(); ++modelIndex) {
            loadModel(modelIndex, models[modelIndex]);
        }
        if (mTextureSources.empty())
            mTextureSources.push_back(cm::AbsLoader::emptyTexture(cm::Texture::BASE_COLOR));

        const std::string skyboxPath = mJson.value("skybox", std::string());
        if (!skyboxPath.empty()) {
            const cm::Texture skybox = cm::AbsLoader::loadSkybox(ASSETS_PATH + skyboxPath);
            mTextureSources.push_back(skybox);
            mSkyboxIndex = static_cast<uint32_t>(mTextureSources.size() - 1);
            mSkyboxName = fs::path(skyboxPath).filename().string();
            mSkyboxPath = skyboxPath;
        }

        if (mExplicit) {
            applyResolvedMaterials();
            loadExplicitInstances();
        }

        recomputeEmissiveIndices();
    }

    void Scene::loadModel(const uint32_t modelIndex, const std::string &path) {
        cu::Timer timer;
        timer.start();
        auto loader = new cm::Loader;
        loader->setModel(ASSETS_PATH + path);
        loader->load(glm::mat4(1.0f));
        INFO << "Model (" << fs::path(path).filename().stem().string() << ") load time: " << timer.duration() / 1000 << " sec";
        for (size_t meshIndex = 0; meshIndex < loader->meshes().size(); ++meshIndex) {
            const auto &mesh = loader->meshes()[meshIndex];
            std::vector<Vertex> vertices{};
            vertices.reserve(mesh.numVertices);
            for (size_t i = 0; i < mesh.numVertices; ++i) {
                const cm::Vertex &modelVertex = loader->vertices()[mesh.baseVertex + i];
                Vertex vertex{
                    .pos = modelVertex.pos,
                    .texCoord = modelVertex.texCoord0,
                    .normal = modelVertex.normal,
                    .tangent = modelVertex.tangent,
                };
                vertices.push_back(vertex);
            }
            std::vector<uint32_t> indices{};
            indices.reserve(mesh.numIndices);
            for (size_t i = 0; i < mesh.numIndices; ++i) {
                indices.push_back(loader->indices()[mesh.baseIndex + i]);
            }
            float area = 0;
            std::vector<float> triAreas{};
            triAreas.reserve(indices.size() / 3);
            for (size_t i = 0; i < indices.size(); i += 3) {
                Vertex v0 = vertices[indices[i + 0]];
                Vertex v1 = vertices[indices[i + 1]];
                Vertex v2 = vertices[indices[i + 2]];
                const float triArea = 0.5f * glm::length(glm::cross(v1.pos - v0.pos, v2.pos - v0.pos));
                triAreas.push_back(triArea);
                area += triArea;
            }
            glm::vec3 aabbMin(std::numeric_limits<float>::max());
            glm::vec3 aabbMax(std::numeric_limits<float>::lowest());
            for (const Vertex &vertex: vertices) {
                aabbMin = glm::min(aabbMin, vertex.pos);
                aabbMax = glm::max(aabbMax, vertex.pos);
            }

            mMeshes.emplace_back();
            MeshData& meshData = mMeshes.back();
            meshData.area = area;
            meshData.modelIndex = modelIndex;
            meshData.meshName = mesh.name;
            meshData.bbox = {aabbMin, aabbMax};
            meshData.triAreas = std::move(triAreas);
            meshData.vertices = std::move(vertices);
            meshData.indices = std::move(indices);
            meshData.indexCount = static_cast<uint32_t>(meshData.indices.size());
            if (mExplicit) continue;

            auto allInstances = mJson["instances"];
            decltype(allInstances) jsonInstances;
            for (const auto &instance: allInstances) {
                if (instance["modelIndex"] != modelIndex) continue;
                jsonInstances.push_back(instance);
            }

            uint32_t baseMaterial = mMaterials.size();
            for (const auto &instance: jsonInstances) {
                glm::vec3 rot = toVec3(instance["localRotation"]);
                Transform transform;
                transform.position = toVec3(instance["localPosition"]);
                transform.scale = toVec3(instance["localScale"]);
                glm::quat qx = glm::angleAxis(glm::radians(rot.x), glm::vec3(1, 0, 0));
                glm::quat qy = glm::angleAxis(glm::radians(rot.y), glm::vec3(0, 1, 0));
                glm::quat qz = glm::angleAxis(glm::radians(rot.z), glm::vec3(0, 0, 1));
                transform.rotation = glm::normalize(qy * qx * qz);
                uint32_t materialIndex = instance["texIndex"];
                if (materialIndex == UINT32_MAX) materialIndex = baseMaterial + mesh.materialIndex;
                InstanceData instanceData{
                    .name = instance["name"],
                    .meshName = mesh.name,
                    .transform = transform,
                    .meshIndex = static_cast<uint32_t>(mMeshes.size() - 1),
                    .materialIndex = materialIndex,
                    .indexCount = meshData.indexCount
                };
                mInstances.push_back(instanceData);
            }

        }
        mMaterials.reserve(mMaterials.size() + loader->materials().size());
        for (const auto &loaderMaterial: loader->materials()) {
            glm::vec3 emission = glm::vec3(loaderMaterial.emissiveColor) * loaderMaterial.emissiveStrength;
            float emissionLum = std::max(emission.r, std::max(emission.g, emission.b));
            Material material{
                .name = loaderMaterial.mName.empty() ? "Unknown" : loaderMaterial.mName,
                .baseColor = loaderMaterial.diffuseColor,
                .luminance = emissionLum,
                .metalness = loaderMaterial.metallic,
                .roughness = loaderMaterial.roughness,
                .ior = loaderMaterial.ior,
                .specular = loaderMaterial.specularFactor,
                .transmission = loaderMaterial.transmission,
                .clearcoat = loaderMaterial.clearcoat,
                .clearcoatRoughness = loaderMaterial.clearcoatRoughness,
                .opacity = 1.0f - loaderMaterial.mTransparencyFactor,
            };
            const cm::Texture &baseColorTexture = loaderMaterial.mTextures[cm::Texture::BASE_COLOR];
            const cm::Texture &normalTexture = loaderMaterial.mTextures[cm::Texture::NORMAL];
            const cm::Texture &metalRoughnessTexture = loaderMaterial.mTextures[cm::Texture::METAL_ROUGHNESS];
            if (!baseColorTexture.empty()) {
                mTextureSources.push_back(baseColorTexture);
                material.baseColorTexIndex = mTextureSources.size() - 1;
                material.baseColorTexName = baseColorTexture.mName;
            }
            if (!normalTexture.empty()) {
                mTextureSources.push_back(normalTexture);
                material.normalTexIndex = mTextureSources.size() - 1;
                material.normalTexName = normalTexture.mName;
            }
            if (!metalRoughnessTexture.empty()) {
                mTextureSources.push_back(metalRoughnessTexture);
                material.metalRoughnessTexIndex = mTextureSources.size() - 1;
                material.metalRoughnessTexName = metalRoughnessTexture.mName;
            }
            const cm::Texture &clearcoatTexture = loaderMaterial.mTextures[cm::Texture::CLEARCOAT];
            const cm::Texture &clearcoatRoughnessTexture = loaderMaterial.mTextures[cm::Texture::CLEARCOAT_ROUGHNESS];
            if (!clearcoatTexture.empty()) {
                mTextureSources.push_back(clearcoatTexture);
                material.clearcoatTexIndex = mTextureSources.size() - 1;
                material.clearcoatTexName = clearcoatTexture.mName;
            }
            if (!clearcoatRoughnessTexture.empty()) {
                mTextureSources.push_back(clearcoatRoughnessTexture);
                material.clearcoatRoughnessTexIndex = mTextureSources.size() - 1;
                material.clearcoatRoughnessTexName = clearcoatRoughnessTexture.mName;
            }
            mMaterials.push_back(material);
        }
    }

    void Scene::loadMaterials() {
        auto materials = mJson["materials"];
        mMaterials.resize(materials.size());
        for (int materialIndex = 0; materialIndex < materials.size(); ++materialIndex) {
            Material& material = mMaterials[materialIndex];
            auto jsonMaterial = materials[materialIndex];
            material = {
                .name = jsonMaterial["name"],
                .baseColor = toVec3(jsonMaterial["color"]),
                .luminance = jsonMaterial["luminance"],
                .metalness = jsonMaterial.value("metalness", 0.0f),
                .roughness = jsonMaterial.value("roughness", 0.0f),
                .ior = jsonMaterial.value("ior", 1.5f),
                .specular = jsonMaterial.value("specular", 0.0f),
                .transmission = jsonMaterial.value("transmission", 0.0f),
                .clearcoat = jsonMaterial.value("clearcoat", 0.0f),
                .clearcoatRoughness = jsonMaterial.value("clearcoatRoughness", 0.0f),
                .absorption = jsonMaterial.contains("absorption") && jsonMaterial["absorption"].is_array() ? toVec3(jsonMaterial["absorption"]) : glm::vec3(1.0f),
                .opacity = jsonMaterial.value("opacity", 1.0f),
                .normalScale = jsonMaterial.value("normalScale", 1.0f),
                .anisotropy = jsonMaterial.value("anisotropy", 0.0f),
                .sheen = jsonMaterial.value("sheen", 0.0f),
                .translucency = jsonMaterial.value("translucency", 0.0f)
            };
        }
    }

    void Scene::applyResolvedMaterials() {
        if (!mJson.contains("materialsResolved")) return;
        const auto& resolved = mJson["materialsResolved"];
        if (resolved.size() > mMaterials.size()) mMaterials.resize(resolved.size());
        for (size_t i = 0; i < resolved.size(); ++i) {
            const auto& jm = resolved[i];
            Material& material = mMaterials[i];
            material.name = jm.value("name", material.name);
            material.baseColor = toVec3(jm["color"]);
            material.luminance = jm["luminance"];
            material.metalness = jm.value("metalness", material.metalness);
            material.roughness = jm.value("roughness", material.roughness);
            material.ior          = jm.value("ior", material.ior);
            material.specular     = jm.value("specular", material.specular);
            material.transmission = jm.value("transmission", material.transmission);
            material.clearcoat          = jm.value("clearcoat", material.clearcoat);
            material.clearcoatRoughness = jm.value("clearcoatRoughness", material.clearcoatRoughness);
            if (jm.contains("absorption") && jm["absorption"].is_array()) material.absorption = toVec3(jm["absorption"]);
            material.opacity = jm.value("opacity", material.opacity);
            material.normalScale = jm.value("normalScale", material.normalScale);
            material.anisotropy = jm.value("anisotropy", material.anisotropy);
            material.sheen = jm.value("sheen", material.sheen);
            material.translucency = jm.value("translucency", material.translucency);

            loadResolvedTexture(jm, "baseColorTex", cm::Texture::BASE_COLOR,
                material.baseColorTexIndex, material.baseColorTexName, material.baseColorTexPath);
            loadResolvedTexture(jm, "normalTex", cm::Texture::NORMAL,
                material.normalTexIndex, material.normalTexName, material.normalTexPath);
            loadResolvedTexture(jm, "metalRoughnessTex", cm::Texture::METAL_ROUGHNESS,
                material.metalRoughnessTexIndex, material.metalRoughnessTexName, material.metalRoughnessTexPath);
            loadResolvedTexture(jm, "clearcoatTex", cm::Texture::CLEARCOAT,
                material.clearcoatTexIndex, material.clearcoatTexName, material.clearcoatTexPath);
            loadResolvedTexture(jm, "clearcoatRoughnessTex", cm::Texture::CLEARCOAT_ROUGHNESS,
                material.clearcoatRoughnessTexIndex, material.clearcoatRoughnessTexName, material.clearcoatRoughnessTexPath);
        }
    }

    void Scene::loadResolvedTexture(const json& jm, const char* key, int textureType,
                                          uint32_t& texIndex, std::string& texName, std::string& texPath) {
        if (!jm.contains(key)) return;
        const std::string rel = jm[key];
        const std::string full = (fs::path(ASSETS_PATH) / rel).string();
        std::error_code ec;
        if (!fs::exists(full, ec)) {
            WARNING << "Scene texture not found: " << full;
            return;
        }
        mTextureSources.push_back(
            cm::AbsLoader::loadTexture(full, static_cast<cm::Texture::Type>(textureType)));
        texIndex = static_cast<uint32_t>(mTextureSources.size() - 1);
        texName  = fs::path(rel).filename().string();
        texPath  = rel;
    }

    void Scene::loadExplicitInstances() {
        mInstances.clear();
        for (const auto& ji : mJson["instances"]) {
            const uint32_t meshIndex = ji["meshIndex"];
            if (meshIndex >= mMeshes.size()) continue;
            Transform transform;
            transform.position = toVec3(ji["localPosition"]);
            transform.scale = toVec3(ji["localScale"]);
            const auto& q = ji["rotation"];
            transform.rotation = glm::quat(q[3].get<float>(), q[0].get<float>(),
                                           q[1].get<float>(), q[2].get<float>());
            InstanceData instanceData{
                .name = ji["name"],
                .meshName = ji.value("meshName", mMeshes[meshIndex].meshName),
                .transform = transform,
                .meshIndex = meshIndex,
                .materialIndex = ji["materialIndex"],
                .indexCount = mMeshes[meshIndex].indexCount
            };
            mInstances.push_back(instanceData);
        }
    }

    json Scene::save() const {
        json scene = mJson;
        scene["version"] = 2;
        scene["directLight"]["direction"] = { mDirectLight.dir.x, mDirectLight.dir.y, mDirectLight.dir.z };
        scene["directLight"]["intensity"] = mDirectLight.intensity;
        scene["skyColor"] = { mSkyColor.r, mSkyColor.g, mSkyColor.b };

        if (mSkyboxIndex != UINT32_MAX) scene["skybox"] = mSkyboxPath;
        else scene.erase("skybox");

        json materials = json::array();
        for (const auto& material : mMaterials) {
            json jm;
            jm["name"]          = material.name;
            jm["color"]         = { material.baseColor.r, material.baseColor.g, material.baseColor.b };
            jm["luminance"]     = material.luminance;
            jm["metalness"]     = material.metalness;
            jm["roughness"]     = material.roughness;
            jm["ior"]           = material.ior;
            jm["specular"]      = material.specular;
            jm["transmission"]  = material.transmission;
            jm["clearcoat"]            = material.clearcoat;
            jm["clearcoatRoughness"]   = material.clearcoatRoughness;
            jm["absorption"]           = { material.absorption.r, material.absorption.g, material.absorption.b };
            jm["opacity"]              = material.opacity;
            jm["normalScale"]          = material.normalScale;
            jm["anisotropy"]           = material.anisotropy;
            jm["sheen"]                = material.sheen;
            if (material.translucency > 0.0f) jm["translucency"] = material.translucency;
            if (!material.baseColorTexPath.empty()) jm["baseColorTex"] = material.baseColorTexPath;
            if (!material.normalTexPath.empty()) jm["normalTex"] = material.normalTexPath;
            if (!material.metalRoughnessTexPath.empty()) jm["metalRoughnessTex"] = material.metalRoughnessTexPath;
            if (!material.clearcoatTexPath.empty()) jm["clearcoatTex"] = material.clearcoatTexPath;
            if (!material.clearcoatRoughnessTexPath.empty()) jm["clearcoatRoughnessTex"] = material.clearcoatRoughnessTexPath;
            materials.push_back(jm);
        }
        scene["materialsResolved"] = materials;

        json instances = json::array();
        for (const auto& instance : mInstances) {
            const Transform& t = instance.transform;
            const glm::vec3 euler = glm::degrees(glm::eulerAngles(t.rotation));
            json ji;
            ji["name"]          = instance.name;
            ji["meshName"]      = instance.meshName;
            ji["modelIndex"]    = mMeshes[instance.meshIndex].modelIndex;
            ji["meshIndex"]     = instance.meshIndex;
            ji["materialIndex"] = instance.materialIndex;
            ji["localPosition"] = { t.position.x, t.position.y, t.position.z };
            ji["localRotation"] = { euler.x, euler.y, euler.z };
            ji["localScale"]    = { t.scale.x, t.scale.y, t.scale.z };
            ji["rotation"]      = { t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w };
            instances.push_back(ji);
        }
        scene["instances"] = instances;
        return scene;
    }

    uint32_t Scene::addMaterial(const Material& material) {
        mMaterials.push_back(material);
        return static_cast<uint32_t>(mMaterials.size() - 1);
    }

    void Scene::setMaterial(const uint32_t index, const Material& material) {
        if (index >= mMaterials.size()) return;
        mMaterials[index] = material;
    }

    void Scene::setMaterialTexture(const uint32_t materialIndex, const int textureType,
                                   const uint32_t texIndex, const std::string& name, const std::string& path) {
        if (materialIndex >= mMaterials.size()) return;
        Material& material = mMaterials[materialIndex];
        switch (textureType) {
            case 1:  material.normalTexIndex = texIndex; material.normalTexName = name; material.normalTexPath = path; break;
            case 2:  material.metalRoughnessTexIndex = texIndex; material.metalRoughnessTexName = name; material.metalRoughnessTexPath = path; break;
            case 3:  material.clearcoatTexIndex = texIndex; material.clearcoatTexName = name; material.clearcoatTexPath = path; break;
            case 4:  material.clearcoatRoughnessTexIndex = texIndex; material.clearcoatRoughnessTexName = name; material.clearcoatRoughnessTexPath = path; break;
            default: material.baseColorTexIndex = texIndex; material.baseColorTexName = name; material.baseColorTexPath = path; break;
        }
    }

    uint32_t Scene::addTextureSource(cm::Texture texture) {
        mTextureSources.push_back(std::move(texture));
        return static_cast<uint32_t>(mTextureSources.size() - 1);
    }

    void Scene::addInstance(const InstanceData& instance) {
        mInstances.push_back(instance);
    }

    void Scene::removeInstance(const uint32_t index) {
        if (index >= mInstances.size()) return;
        mInstances.erase(mInstances.begin() + index);
    }

    void Scene::setInstanceTransform(const uint32_t index, const Transform& transform) {
        if (index >= mInstances.size()) return;
        mInstances[index].transform = transform;
    }

    void Scene::setInstanceMaterial(const uint32_t instanceIndex, const uint32_t materialIndex) {
        if (instanceIndex >= mInstances.size() || materialIndex >= mMaterials.size()) return;
        mInstances[instanceIndex].materialIndex = materialIndex;
    }

    void Scene::setSkybox(const uint32_t index, const std::string& name, const std::string& path) {
        mSkyboxIndex = index;
        mSkyboxName  = name;
        mSkyboxPath  = path;
    }

    void Scene::clearSkybox() {
        mSkyboxIndex = UINT32_MAX;
        mSkyboxName.clear();
        mSkyboxPath.clear();
    }

    void Scene::setSkyColor(const glm::vec3& color) { mSkyColor = color; }

    void Scene::setDirectLight(const DirectLight& light) { mDirectLight = light; }

    void Scene::recomputeEmissiveIndices() {
        mEmissiveIndices.clear();
        for (uint32_t i = 0; i < mInstances.size(); ++i) {
            if (mMaterials[mInstances[i].materialIndex].luminance == 0) continue;
            mEmissiveIndices.push_back(i);
        }
    }
}