//
// Created by igor on 6/10/26.
//

#include "Model/Scene.hpp"

#include <algorithm>
#include <utility>
#include <glm/gtx/matrix_decompose.hpp>

namespace {
    std::string relativeToAssets(const std::string& path) {
        const std::string assets = ASSETS_PATH;
        return path.rfind(assets, 0) == 0 ? path.substr(assets.size()) : path;
    }

    std::string textureKey(const std::string& relPath, const int type) {
        return relPath + "|" + std::to_string(type);
    }
}

namespace crv::graphics::vulkan {
    Scene::Scene(const SceneCreateInfo&) {}

    void Scene::load(const json& scene) {
        mJson = scene;
        mVersion = mJson.value("version", 1);
        auto directLight = mJson["directLight"];
        mDirectLight.dir = glm::vec4(toVec3(directLight["direction"]), 1);
        mDirectLight.intensity = directLight["intensity"];
        if (mJson.contains("skyColor") && mJson["skyColor"].is_array()) mSkyColor = toVec3(mJson["skyColor"]);

        if (mVersion < 2 || mJson.contains("materialsResolved")) loadJsonMaterials();
        const std::vector<std::string> models = mJson.value("modelImports", std::vector<std::string>{});
        for (int modelIndex = 0; modelIndex < models.size(); ++modelIndex) {
            loadModel(modelIndex, models[modelIndex]);
        }
        if (mTextureSources.empty())
            mTextureSources.push_back(cm::AbsLoader::emptyTexture(cm::Texture::BASE_COLOR));

        const std::string skyboxPath = mJson.value("skybox", std::string());
        if (!skyboxPath.empty()) {
            mSkyboxIndex = addTextureSource(cm::AbsLoader::loadSkybox(ASSETS_PATH + skyboxPath));
            mSkyboxPath = skyboxPath;
        }

        if (mVersion >= 2) {
            applyResolvedMaterials();
            loadExplicitInstances();
        }

        recomputeEmissiveIndices();
        recomputeWorlds();
    }

    void Scene::loadModel(const uint32_t modelIndex, const std::string &path) {
        cu::Timer timer;
        timer.start();
        cm::Loader loader;
        loader.setModel(ASSETS_PATH + path);
        loader.load(glm::mat4(1.0f));
        INFO << "Model (" << fs::path(path).filename().stem().string() << ") load time: " << timer.duration() / 1000 << " sec";

        const auto meshBase     = static_cast<uint32_t>(mMeshes.size());
        const auto materialBase = static_cast<uint32_t>(mMaterials.size());
        buildMeshes(loader, modelIndex);
        if (mVersion < 2) buildInstances(loader, modelIndex, meshBase, materialBase);
        loadModelMaterials(loader);
    }

    void Scene::buildMeshes(cm::Loader& loader, const uint32_t modelIndex) {
        for (size_t meshIndex = 0; meshIndex < loader.meshes().size(); ++meshIndex) {
            const auto &mesh = loader.meshes()[meshIndex];
            std::vector<Vertex> vertices{};
            vertices.reserve(mesh.numVertices);
            for (size_t i = 0; i < mesh.numVertices; ++i) {
                const cm::Vertex &modelVertex = loader.vertices()[mesh.baseVertex + i];
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
                indices.push_back(loader.indices()[mesh.baseIndex + i]);
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

            const auto globalIndex = static_cast<uint32_t>(mMeshes.size());
            mMeshes.emplace_back();
            MeshData& meshData = mMeshes.back();
            meshData.area = area;
            meshData.modelIndex = modelIndex;
            meshData.meshName = mesh.name.empty() ? "mesh_" + std::to_string(globalIndex) : mesh.name;
            meshData.bbox = {aabbMin, aabbMax};
            meshData.triAreas = std::move(triAreas);
            meshData.vertices = std::move(vertices);
            meshData.indices = std::move(indices);
            meshData.indexCount = static_cast<uint32_t>(meshData.indices.size());
        }
    }

    void Scene::buildInstances(cm::Loader& loader, const uint32_t modelIndex,
                               const uint32_t meshBase, const uint32_t materialBase) {
        if (!mJson.contains("instances")) return;
        for (const auto &instance: mJson["instances"]) {
            if (instance["modelIndex"] != modelIndex) continue;
            glm::vec3 rot = toVec3(instance["localRotation"]);
            Transform placement;
            placement.position = toVec3(instance["localPosition"]);
            placement.scale = toVec3(instance["localScale"]);
            glm::quat qx = glm::angleAxis(glm::radians(rot.x), glm::vec3(1, 0, 0));
            glm::quat qy = glm::angleAxis(glm::radians(rot.y), glm::vec3(0, 1, 0));
            glm::quat qz = glm::angleAxis(glm::radians(rot.z), glm::vec3(0, 0, 1));
            placement.rotation = glm::normalize(qy * qx * qz);
            const uint32_t materialOverride = instance["texIndex"];

            const auto placementIndex = static_cast<int32_t>(mInstances.size());
            mInstances.push_back(InstanceData{
                .name = instance["name"],
                .transform = placement,
                .parentIndex = -1,
                .meshIndex = InstanceData::NO_MESH,
            });
            addNode(loader.root(), placementIndex, meshBase, materialBase, loader, materialOverride);
        }
    }

    void Scene::addNode(const cm::Node& node, const int32_t parentIndex, const uint32_t meshBase,
                        const uint32_t materialBase, cm::Loader& loader, const uint32_t materialOverride,
                        const glm::mat4& accum) {
        const glm::mat4 localMatrix = accum * node.transform;

        if (node.meshes.empty() && node.children.size() == 1) {
            addNode(node.children[0], parentIndex, meshBase, materialBase, loader, materialOverride, localMatrix);
            return;
        }

        const auto isGeneric = [](const std::string& n) { return n.empty() || n.rfind("_gltfNode", 0) == 0; };
        const Transform local = decomposeTransform(localMatrix);
        auto materialFor = [&](const uint32_t localMeshIndex) {
            return materialOverride != UINT32_MAX
                ? materialOverride
                : materialBase + static_cast<uint32_t>(loader.meshes()[localMeshIndex].materialIndex);
        };

        if (node.meshes.size() == 1 && node.children.empty()) {
            const uint32_t localIndex = node.meshes[0];
            const uint32_t mesh = meshBase + localIndex;
            mInstances.push_back(InstanceData{
                .name = isGeneric(node.name) ? mMeshes[mesh].meshName : node.name,
                .meshName = mMeshes[mesh].meshName,
                .transform = local,
                .parentIndex = parentIndex,
                .meshIndex = mesh,
                .materialIndex = materialFor(localIndex),
                .indexCount = mMeshes[mesh].indexCount
            });
            return;
        }

        const auto groupIndex = static_cast<int32_t>(mInstances.size());
        mInstances.push_back(InstanceData{
            .name = isGeneric(node.name) ? "Group" : node.name,
            .transform = local,
            .parentIndex = parentIndex,
            .meshIndex = InstanceData::NO_MESH,
        });
        for (const uint32_t localIndex : node.meshes) {
            const uint32_t mesh = meshBase + localIndex;
            mInstances.push_back(InstanceData{
                .name = mMeshes[mesh].meshName,
                .meshName = mMeshes[mesh].meshName,
                .transform = {},
                .parentIndex = groupIndex,
                .meshIndex = mesh,
                .materialIndex = materialFor(localIndex),
                .indexCount = mMeshes[mesh].indexCount
            });
        }
        for (const auto& child : node.children)
            addNode(child, groupIndex, meshBase, materialBase, loader, materialOverride);
    }

    void Scene::loadModelMaterials(cm::Loader& loader) {
        mMaterials.reserve(mMaterials.size() + loader.materials().size());
        for (const auto &loaderMaterial: loader.materials()) {
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
            const auto ingest = [&](const cm::Texture::Type type, uint32_t& texIndex, std::string& texPath) {
                const cm::Texture& texture = loaderMaterial.mTextures[type];
                if (texture.empty()) return;
                texIndex = addTextureSource(texture);
                texPath  = relativeToAssets(texture.mPath);
            };
            ingest(cm::Texture::BASE_COLOR, material.baseColorTexIndex, material.baseColorTexPath);
            ingest(cm::Texture::NORMAL, material.normalTexIndex, material.normalTexPath);
            ingest(cm::Texture::METAL_ROUGHNESS, material.metalRoughnessTexIndex, material.metalRoughnessTexPath);
            ingest(cm::Texture::CLEARCOAT, material.clearcoatTexIndex, material.clearcoatTexPath);
            ingest(cm::Texture::CLEARCOAT_ROUGHNESS, material.clearcoatRoughnessTexIndex, material.clearcoatRoughnessTexPath);
            mMaterials.push_back(material);
        }
    }

    void Scene::loadJsonMaterials() {
        if (!mJson.contains("materials")) return;
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
                .translucency = jsonMaterial.value("translucency", 0.0f),
                .thin = jsonMaterial.value("thin", 0.0f)
            };
        }
    }

    void Scene::inheritModelTextures(const std::string& name, const std::vector<Material>& loaded,
                                     std::vector<bool>& taken, Material& material) const {
        for (size_t i = 0; i < loaded.size(); ++i) {
            if (taken[i] || loaded[i].name != name) continue;
            taken[i] = true;
            const Material& base = loaded[i];
            material.name = base.name;
            material.baseColorTexIndex = base.baseColorTexIndex; material.baseColorTexPath = base.baseColorTexPath;
            material.normalTexIndex = base.normalTexIndex; material.normalTexPath = base.normalTexPath;
            material.metalRoughnessTexIndex = base.metalRoughnessTexIndex; material.metalRoughnessTexPath = base.metalRoughnessTexPath;
            material.clearcoatTexIndex = base.clearcoatTexIndex; material.clearcoatTexPath = base.clearcoatTexPath;
            material.clearcoatRoughnessTexIndex = base.clearcoatRoughnessTexIndex; material.clearcoatRoughnessTexPath = base.clearcoatRoughnessTexPath;
            return;
        }
    }

    void Scene::applyResolvedMaterials() {
        const char* key = mJson.contains("materialsResolved") ? "materialsResolved" : "materials";
        if (!mJson.contains(key)) return;
        const auto& resolved = mJson[key];

        const std::vector<Material> loaded = std::move(mMaterials);
        mMaterials.assign(std::max(resolved.size(), loaded.size()), Material{});
        for (size_t i = resolved.size(); i < loaded.size(); ++i) mMaterials[i] = loaded[i];

        std::vector<bool> taken(loaded.size(), false);
        for (size_t i = 0; i < resolved.size(); ++i) {
            const auto& jm = resolved[i];
            Material& material = mMaterials[i];

            if (mVersion < 3) inheritModelTextures(jm.value("name", std::string()), loaded, taken, material);

            const auto get = [&jm](const char* key, auto& field) { field = jm.value(key, field); };
            const auto getVec3 = [&jm](const char* key, glm::vec3& field) {
                if (jm.contains(key) && jm[key].is_array()) field = toVec3(jm[key]);
            };

            get("name", material.name);
            getVec3("color", material.baseColor);
            get("luminance", material.luminance);
            get("metalness", material.metalness);
            get("roughness", material.roughness);
            get("ior", material.ior);
            get("specular", material.specular);
            get("transmission", material.transmission);
            get("clearcoat", material.clearcoat);
            get("clearcoatRoughness", material.clearcoatRoughness);
            getVec3("absorption", material.absorption);
            get("opacity", material.opacity);
            get("normalScale", material.normalScale);
            get("anisotropy", material.anisotropy);
            get("sheen", material.sheen);
            get("translucency", material.translucency);
            get("thin", material.thin);

            loadResolvedTexture(jm, "baseColorTex", cm::Texture::BASE_COLOR,
                material.baseColorTexIndex, material.baseColorTexPath);
            loadResolvedTexture(jm, "normalTex", cm::Texture::NORMAL,
                material.normalTexIndex, material.normalTexPath);
            loadResolvedTexture(jm, "metalRoughnessTex", cm::Texture::METAL_ROUGHNESS,
                material.metalRoughnessTexIndex, material.metalRoughnessTexPath);
            loadResolvedTexture(jm, "clearcoatTex", cm::Texture::CLEARCOAT,
                material.clearcoatTexIndex, material.clearcoatTexPath);
            loadResolvedTexture(jm, "clearcoatRoughnessTex", cm::Texture::CLEARCOAT_ROUGHNESS,
                material.clearcoatRoughnessTexIndex, material.clearcoatRoughnessTexPath);
        }
    }

    void Scene::loadResolvedTexture(const json& jm, const char* key, const int textureType,
                                    uint32_t& texIndex, std::string& texPath) {
        if (!jm.contains(key) || !jm[key].is_string()) return;
        const std::string rel = jm[key];
        if (rel.empty()) {
            texIndex = UINT32_MAX;
            texPath.clear();
            return;
        }

        const auto cached = mTextureByPath.find(textureKey(rel, textureType));
        if (cached != mTextureByPath.end()) {
            texIndex = cached->second;
            texPath  = rel;
            return;
        }

        const std::string full = (fs::path(ASSETS_PATH) / rel).string();
        std::error_code ec;
        if (!fs::exists(full, ec)) {
            WARNING << "Scene texture not found: " << full;
            return;
        }
        texIndex = addTextureSource(cm::AbsLoader::loadTexture(full, static_cast<cm::Texture::Type>(textureType)));
        texPath  = rel;
    }

    void Scene::loadExplicitInstances() {
        mInstances.clear();
        if (!mJson.contains("instances")) return;
        for (const auto& ji : mJson["instances"]) {
            Transform transform;
            if (ji.contains("position")) transform.position = toVec3(ji["position"]);
            if (ji.contains("scale"))    transform.scale    = toVec3(ji["scale"]);
            if (ji.contains("rotation")) {
                const auto& r = ji["rotation"];
                if (r.size() == 4)
                    transform.rotation = glm::quat(r[3].get<float>(), r[0].get<float>(),
                                                   r[1].get<float>(), r[2].get<float>());
                else
                    transform.rotation = glm::normalize(glm::quat(glm::radians(toVec3(r))));
            }

            InstanceData instance;
            instance.name = ji.value("name", std::string());
            instance.transform = transform;
            instance.parentIndex = ji.value("parent", -1);
            if (ji.contains("mesh") && ji["mesh"].get<uint32_t>() < mMeshes.size()) {
                const uint32_t mesh = ji["mesh"];
                const uint32_t material = ji.value("material", 0u);
                if (material >= mMaterials.size())
                    WARNING << "Instance '" << instance.name << "' references material "
                            << material << " of " << mMaterials.size() << "; using 0";
                instance.meshIndex = mesh;
                instance.meshName = mMeshes[mesh].meshName;
                instance.materialIndex = material < mMaterials.size() ? material : 0u;
                instance.indexCount = mMeshes[mesh].indexCount;
            } else {
                instance.meshIndex = InstanceData::NO_MESH;
            }
            mInstances.push_back(instance);
        }
    }

    json Scene::save() const {
        const auto put = [](json& j, const char* key, const auto& value, const auto& fallback) {
            if (value != fallback) j[key] = value;
        };
        const auto putVec3 = [](json& j, const char* key, const glm::vec3& value, const glm::vec3& fallback) {
            if (value != fallback) j[key] = toJson(value);
        };
        const auto putPath = [](json& j, const char* key, const std::string& path) {
            if (!path.empty()) j[key] = path;
        };

        json scene = mJson;
        scene["version"] = 3;
        scene["directLight"]["direction"] = toJson(mDirectLight.dir);
        scene["directLight"]["intensity"] = mDirectLight.intensity;
        scene["skyColor"] = toJson(mSkyColor);

        if (mSkyboxIndex != UINT32_MAX) scene["skybox"] = mSkyboxPath;
        else scene.erase("skybox");

        static const Material def{};
        json materials = json::array();
        for (const auto& m : mMaterials) {
            json jm;
            jm["name"] = m.name;
            putVec3(jm, "color", m.baseColor, def.baseColor);
            put(jm, "luminance", m.luminance, def.luminance);
            put(jm, "metalness", m.metalness, def.metalness);
            put(jm, "roughness", m.roughness, def.roughness);
            put(jm, "ior", m.ior, def.ior);
            put(jm, "specular", m.specular, def.specular);
            put(jm, "transmission", m.transmission, def.transmission);
            put(jm, "clearcoat", m.clearcoat, def.clearcoat);
            put(jm, "clearcoatRoughness", m.clearcoatRoughness, def.clearcoatRoughness);
            putVec3(jm, "absorption", m.absorption, def.absorption);
            put(jm, "opacity", m.opacity, def.opacity);
            put(jm, "normalScale", m.normalScale, def.normalScale);
            put(jm, "anisotropy", m.anisotropy, def.anisotropy);
            put(jm, "sheen", m.sheen, def.sheen);
            put(jm, "translucency", m.translucency, def.translucency);
            put(jm, "thin", m.thin, def.thin);
            putPath(jm, "baseColorTex", m.baseColorTexPath);
            putPath(jm, "normalTex", m.normalTexPath);
            putPath(jm, "metalRoughnessTex", m.metalRoughnessTexPath);
            putPath(jm, "clearcoatTex", m.clearcoatTexPath);
            putPath(jm, "clearcoatRoughnessTex", m.clearcoatRoughnessTexPath);
            materials.push_back(jm);
        }
        scene["materials"] = materials;
        scene.erase("materialsResolved");

        static const Transform defTransform{};
        json instances = json::array();
        for (const auto& instance : mInstances) {
            const Transform& t = instance.transform;
            json ji;
            ji["name"] = instance.name;
            if (instance.parentIndex >= 0) ji["parent"] = instance.parentIndex;
            putVec3(ji, "position", t.position, defTransform.position);
            if (t.rotation != defTransform.rotation)
                ji["rotation"] = toJson(glm::degrees(glm::eulerAngles(t.rotation)));
            putVec3(ji, "scale", t.scale, defTransform.scale);
            if (!instance.isGroup()) {
                ji["mesh"] = instance.meshIndex;
                ji["material"] = instance.materialIndex;
            }
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
                                   const uint32_t texIndex, const std::string& path) {
        if (materialIndex >= mMaterials.size()) return;
        Material& material = mMaterials[materialIndex];
        switch (textureType) {
            case 1:  material.normalTexIndex = texIndex; material.normalTexPath = path; break;
            case 2:  material.metalRoughnessTexIndex = texIndex; material.metalRoughnessTexPath = path; break;
            case 3:  material.clearcoatTexIndex = texIndex; material.clearcoatTexPath = path; break;
            case 4:  material.clearcoatRoughnessTexIndex = texIndex; material.clearcoatRoughnessTexPath = path; break;
            default: material.baseColorTexIndex = texIndex; material.baseColorTexPath = path; break;
        }
    }

    uint32_t Scene::addTextureSource(cm::Texture texture) {
        const std::string key = texture.mPath.empty()
            ? std::string() : textureKey(relativeToAssets(texture.mPath), texture.mType);
        if (!key.empty()) {
            const auto it = mTextureByPath.find(key);
            if (it != mTextureByPath.end()) return it->second;
        }
        mTextureSources.push_back(std::move(texture));
        const auto index = static_cast<uint32_t>(mTextureSources.size() - 1);
        if (!key.empty()) mTextureByPath.emplace(key, index);
        return index;
    }

    void Scene::addModel(const std::string& path) {
        const uint32_t modelIndex = mJson.contains("modelImports")
            ? static_cast<uint32_t>(mJson["modelImports"].size()) : 0u;
        cm::Loader loader;
        loader.setModel(ASSETS_PATH + path);
        loader.load(glm::mat4(1.0f));

        const auto meshBase     = static_cast<uint32_t>(mMeshes.size());
        const auto materialBase = static_cast<uint32_t>(mMaterials.size());
        buildMeshes(loader, modelIndex);
        loadModelMaterials(loader);

        const auto rootIndex = static_cast<int32_t>(mInstances.size());
        mInstances.push_back(InstanceData{
            .name = fs::path(path).stem().string(),
            .parentIndex = -1,
            .meshIndex = InstanceData::NO_MESH,
        });
        addNode(loader.root(), rootIndex, meshBase, materialBase, loader, UINT32_MAX);

        if (!mJson.contains("modelImports")) mJson["modelImports"] = json::array();
        mJson["modelImports"].push_back(path);

        recomputeEmissiveIndices();
        recomputeWorlds();
    }

    std::vector<bool> Scene::subtreeMask(const std::vector<uint32_t>& roots) const {
        const auto count = static_cast<uint32_t>(mInstances.size());
        std::vector<bool> mask(count, false);
        for (const uint32_t index : roots)
            if (index < count) mask[index] = true;
        for (uint32_t i = 0; i < count; ++i) {
            const int32_t parent = mInstances[i].parentIndex;
            if (parent >= 0 && parent < static_cast<int32_t>(count) && mask[parent]) mask[i] = true;
        }
        return mask;
    }

    std::vector<uint32_t> Scene::duplicateInstances(const std::vector<uint32_t>& indices) {
        const auto count = static_cast<uint32_t>(mInstances.size());
        const std::vector<bool> affected = subtreeMask(indices);

        std::vector<int32_t> remap(count, -1);
        std::vector<uint32_t> createdRoots;
        for (uint32_t i = 0; i < count; ++i) {
            if (!affected[i]) continue;
            InstanceData copy = mInstances[i];
            const auto newIndex = static_cast<int32_t>(mInstances.size());
            remap[i] = newIndex;
            if (copy.parentIndex >= 0 && affected[copy.parentIndex]) {
                copy.parentIndex = remap[copy.parentIndex];
            } else {
                copy.name += "_copy";
                createdRoots.push_back(static_cast<uint32_t>(newIndex));
            }
            mInstances.push_back(copy);
        }
        recomputeWorlds();
        return createdRoots;
    }

    void Scene::removeInstances(const std::vector<uint32_t>& indices) {
        const auto count = static_cast<uint32_t>(mInstances.size());
        const std::vector<bool> removed = subtreeMask(indices);

        std::vector<int32_t> remap(count, -1);
        std::vector<InstanceData> kept;
        kept.reserve(count);
        for (uint32_t i = 0; i < count; ++i) {
            if (removed[i]) continue;
            remap[i] = static_cast<int32_t>(kept.size());
            kept.push_back(mInstances[i]);
        }
        for (InstanceData& instance : kept)
            if (instance.parentIndex >= 0) instance.parentIndex = remap[instance.parentIndex];
        mInstances = std::move(kept);
        recomputeWorlds();
    }

    void Scene::setInstanceName(const uint32_t index, const std::string& name) {
        if (index >= mInstances.size()) return;
        mInstances[index].name = name;
    }

    void Scene::setInstanceTransform(const uint32_t index, const Transform& transform) {
        if (index >= mInstances.size()) return;
        mInstances[index].transform = transform;
        recomputeWorlds();
    }

    void Scene::setInstanceMaterial(const uint32_t instanceIndex, const uint32_t materialIndex) {
        if (instanceIndex >= mInstances.size() || materialIndex >= mMaterials.size()) return;
        mInstances[instanceIndex].materialIndex = materialIndex;
    }

    void Scene::setSkybox(const uint32_t index, const std::string& path) {
        mSkyboxIndex = index;
        mSkyboxPath  = path;
    }

    void Scene::clearSkybox() {
        mSkyboxIndex = UINT32_MAX;
        mSkyboxPath.clear();
    }

    void Scene::setSkyColor(const glm::vec3& color) { mSkyColor = color; }

    void Scene::setDirectLight(const DirectLight& light) { mDirectLight = light; }

    void Scene::recomputeEmissiveIndices() {
        mEmissiveIndices.clear();
        for (uint32_t i = 0; i < mInstances.size(); ++i) {
            if (mInstances[i].isGroup()) continue;
            if (mInstances[i].materialIndex >= mMaterials.size()) continue;
            if (mMaterials[mInstances[i].materialIndex].luminance == 0) continue;
            mEmissiveIndices.push_back(i);
        }
    }

    void Scene::recomputeWorlds() {
        for (auto& instance : mInstances) {
            const glm::mat4 local = instance.transform.matrix();
            instance.world = instance.parentIndex >= 0
                ? mInstances[instance.parentIndex].world * local
                : local;
        }
    }
}