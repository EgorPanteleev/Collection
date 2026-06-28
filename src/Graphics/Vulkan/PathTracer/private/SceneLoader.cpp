//
// Created by igor on 6/10/26.
//

#include "SceneLoader.hpp"

namespace crv::graphics::vulkan {
    namespace {
        std::vector<AliasEntry> buildAliasTable(const std::vector<float>& weights) {
            const size_t n = weights.size();
            std::vector<AliasEntry> table(n);
            double sum = 0.0;
            for (const float w : weights) sum += w;

            // Scale weights so the mean is 1, then split into under-/over-full columns.
            std::vector<double> scaled(n);
            std::vector<uint32_t> small, large;
            small.reserve(n);
            large.reserve(n);
            for (size_t i = 0; i < n; ++i) {
                scaled[i] = sum > 0.0 ? weights[i] * static_cast<double>(n) / sum : 1.0;
                (scaled[i] < 1.0 ? small : large).push_back(static_cast<uint32_t>(i));
            }
            while (!small.empty() && !large.empty()) {
                const uint32_t s = small.back(); small.pop_back();
                const uint32_t l = large.back(); large.pop_back();
                table[s].prob  = static_cast<float>(scaled[s]);
                table[s].alias = l;
                scaled[l] = (scaled[l] + scaled[s]) - 1.0;
                (scaled[l] < 1.0 ? small : large).push_back(l);
            }
            // Remaining columns have probability 1 (alias unused).
            for (const uint32_t l : large) { table[l].prob = 1.0f; table[l].alias = l; }
            for (const uint32_t s : small) { table[s].prob = 1.0f; table[s].alias = s; }
            return table;
        }
    }

    SceneLoader::SceneLoader(const SceneLoaderCreateInfo& info):
    mContext(info.context) {}

    void SceneLoader::loadScene(const json& scene) {
        mJson = scene;
        mExplicit = mJson.value("version", 1) >= 2;
        auto directLight = mJson["directLight"];
        mDirectLight.dir = glm::vec4(toVec3(directLight["direction"]), 1);
        mDirectLight.intensity = directLight["intensity"];

        loadMaterials();
        std::vector<std::string> models = mJson["modelImports"];
        for (int modelIndex = 0; modelIndex < models.size(); ++modelIndex) {
            loadModel(modelIndex, models[modelIndex]);
        }
        if (mTextures.empty())
            mTextures.push_back(toTexture(mContext, cm::AbsLoader::emptyTexture(cm::Texture::BASE_COLOR)));

        const std::string skyboxPath = mJson.value("skybox", std::string());
        if (!skyboxPath.empty()) {
            mTextures.push_back(toTexture(mContext, cm::AbsLoader::loadSkybox(ASSETS_PATH + skyboxPath)));
            mSkyboxIndex = static_cast<uint32_t>(mTextures.size() - 1);
            mSkyboxName = fs::path(skyboxPath).filename().string();
            mSkyboxPath = skyboxPath;
        }

        if (mExplicit) {
            applyResolvedMaterials();
            loadExplicitInstances();
            buildEmissiveAliasTables();
        }

        for (uint32_t i = 0; i < mInstances.size(); ++i) {
            if (mMaterials[mInstances[i].materialIndex].luminance == 0) continue;
            mEmissiveIndices.push_back(i);
        }
    }

    void SceneLoader::loadModel(const uint32_t modelIndex, const std::string &path) {
        cu::Timer timer;
        timer.start();
        auto loader = new cm::Loader;
        loader->setModel(ASSETS_PATH + path);
        loader->load(glm::mat4(1.0f));
        INFO << "Model (" << fs::path(path).filename().stem().string() << ") load time: " << timer.duration() / 1000 << " sec";
        auto [commandBuffer, cmdData] = beginCommandBuffer(mContext->device(),
                                                           mContext->familyIndex(QueueFamilyType::GRAPHICS).value());
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
            mBLASDatas.emplace_back();
            BLASData &blasData = mBLASDatas.back();
            blasData.area = area;
            blasData.indexCount = indices.size();
            blasData.modelIndex = modelIndex;
            blasData.meshName = mesh.name;
            blasData.triAreas = triAreas;
            const size_t verticesSize = sizeof(Vertex) * vertices.size();
            const BufferCreateInfo vertexBufferCreateInfo{
                .allocator = mContext->allocator(),
                .size = verticesSize,
                .bufferUsage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                               VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                               VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                .memoryUsage = VMA_MEMORY_USAGE_GPU_ONLY
            };
            blasData.vertexBuffer = Buffer(vertexBufferCreateInfo);
            const CopyDataToGPUBufferInfo vertexCopyInfo{
                .data = vertices.data(),
                .size = verticesSize,
                .allocator = mContext->allocator(),
                .buffer = blasData.vertexBuffer.get(),
                .device = mContext->device(),
                .queueFamilyIndex = mContext->familyIndex(QueueFamilyType::GRAPHICS).value(),
                .queue = mContext->queue(QueueFamilyType::GRAPHICS)
            };
            Buffer::copy(vertexCopyInfo);

            const size_t indicesSize = sizeof(uint32_t) * indices.size();
            const BufferCreateInfo indexBufferCreateInfo{
                .allocator = mContext->allocator(),
                .size = indicesSize,
                .bufferUsage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                               VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                               VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                .memoryUsage = VMA_MEMORY_USAGE_GPU_ONLY
            };
            blasData.indexBuffer = Buffer(indexBufferCreateInfo);
            const CopyDataToGPUBufferInfo indexCopyInfo{
                .data = indices.data(),
                .size = indicesSize,
                .allocator = mContext->allocator(),
                .buffer = blasData.indexBuffer.get(),
                .device = mContext->device(),
                .queueFamilyIndex = mContext->familyIndex(QueueFamilyType::GRAPHICS).value(),
                .queue = mContext->queue(QueueFamilyType::GRAPHICS)
            };
            Buffer::copy(indexCopyInfo);

            BLASCreateInfo blasCreateInfo{
                .commandBuffer = commandBuffer,
                .device = mContext->device(),
                .physicalDevice = mContext->physicalDevice(),
                .allocator = mContext->allocator(),
                .vertexAddress = blasData.vertexBuffer.deviceAddress(mContext->device()),
                .vertexStride = sizeof(Vertex),
                .vertexCount = static_cast<uint32_t>(vertices.size()),
                .indexAddress = blasData.indexBuffer.deviceAddress(mContext->device()),
                .indexCount = static_cast<uint32_t>(indices.size())
            };
            blasData.blas = AccelerationStructure(blasCreateInfo);
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
                    .meshIndex = static_cast<uint32_t>(mBLASDatas.size() - 1),
                    .materialIndex = materialIndex,
                    .indexCount = static_cast<uint32_t>(indices.size())
                };
                mInstances.push_back(instanceData);
            }

            bool meshEmissive = false;
            for (const auto &instance: jsonInstances) {
                uint32_t materialIndex = instance["texIndex"];
                if (materialIndex == UINT32_MAX) materialIndex = baseMaterial + mesh.materialIndex;
                if (materialIndex < mMaterials.size() && mMaterials[materialIndex].luminance > 0.0f) {
                    meshEmissive = true;
                    break;
                }
            }
            if (meshEmissive) buildAlias(blasData);
        }
        endCommandBuffer(cmdData, mContext->queue(QueueFamilyType::GRAPHICS));
        mMaterials.reserve(mMaterials.size() + loader->materials().size());
        for (const auto &loaderMaterial: loader->materials()) {
            glm::vec3 emission = glm::vec3(loaderMaterial.emissiveColor) * loaderMaterial.emissiveStrength;
            float emissionLum = std::max(emission.r, std::max(emission.g, emission.b));
            Material material{
                .name = loaderMaterial.mName.empty() ? "Unknown" : loaderMaterial.mName,
                .baseColor = loaderMaterial.diffuseColor,
                .emissionColor = emissionLum > 0.0f ? emission / emissionLum : glm::vec3(1.0f),
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
                mTextures.push_back(toTexture(mContext, baseColorTexture));
                material.baseColorTexIndex = mTextures.size() - 1;
                material.baseColorTexName = baseColorTexture.mName;
            }
            if (!normalTexture.empty()) {
                mTextures.push_back(toTexture(mContext, normalTexture));
                material.normalTexIndex = mTextures.size() - 1;
                material.normalTexName = normalTexture.mName;
            }
            if (!metalRoughnessTexture.empty()) {
                mTextures.push_back(toTexture(mContext, metalRoughnessTexture));
                material.metalRoughnessTexIndex = mTextures.size() - 1;
                material.metalRoughnessTexName = metalRoughnessTexture.mName;
            }
            const cm::Texture &clearcoatTexture = loaderMaterial.mTextures[cm::Texture::CLEARCOAT];
            const cm::Texture &clearcoatRoughnessTexture = loaderMaterial.mTextures[cm::Texture::CLEARCOAT_ROUGHNESS];
            if (!clearcoatTexture.empty()) {
                mTextures.push_back(toTexture(mContext, clearcoatTexture));
                material.clearcoatTexIndex = mTextures.size() - 1;
                material.clearcoatTexName = clearcoatTexture.mName;
            }
            if (!clearcoatRoughnessTexture.empty()) {
                mTextures.push_back(toTexture(mContext, clearcoatRoughnessTexture));
                material.clearcoatRoughnessTexIndex = mTextures.size() - 1;
                material.clearcoatRoughnessTexName = clearcoatRoughnessTexture.mName;
            }
            mMaterials.push_back(material);
        }
    }

    void SceneLoader::loadMaterials() {
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
                .opacity = jsonMaterial.value("opacity", 1.0f)
            };
        }
    }

    void SceneLoader::buildAlias(BLASData& blasData) {
        if (blasData.triAreas.empty() || blasData.aliasBuffer.get() != VK_NULL_HANDLE) return;
        auto aliasTable = buildAliasTable(blasData.triAreas);
        const size_t aliasSize = sizeof(AliasEntry) * aliasTable.size();
        const BufferCreateInfo aliasBufferCreateInfo{
            .allocator = mContext->allocator(),
            .size = aliasSize,
            .bufferUsage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .memoryUsage = VMA_MEMORY_USAGE_GPU_ONLY
        };
        blasData.aliasBuffer = Buffer(aliasBufferCreateInfo);
        const CopyDataToGPUBufferInfo aliasCopyInfo{
            .data = aliasTable.data(),
            .size = aliasSize,
            .allocator = mContext->allocator(),
            .buffer = blasData.aliasBuffer.get(),
            .device = mContext->device(),
            .queueFamilyIndex = mContext->familyIndex(QueueFamilyType::GRAPHICS).value(),
            .queue = mContext->queue(QueueFamilyType::GRAPHICS)
        };
        Buffer::copy(aliasCopyInfo);
    }

    void SceneLoader::applyResolvedMaterials() {
        if (!mJson.contains("materialsResolved")) return;
        const auto& resolved = mJson["materialsResolved"];
        for (size_t i = 0; i < resolved.size() && i < mMaterials.size(); ++i) {
            const auto& jm = resolved[i];
            Material& material = mMaterials[i];
            material.baseColor = toVec3(jm["color"]);
            if (jm.contains("emissionColor")) material.emissionColor = toVec3(jm["emissionColor"]);
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
        }
    }

    void SceneLoader::loadExplicitInstances() {
        mInstances.clear();
        for (const auto& ji : mJson["instances"]) {
            const uint32_t meshIndex = ji["meshIndex"];
            if (meshIndex >= mBLASDatas.size()) continue;
            Transform transform;
            transform.position = toVec3(ji["localPosition"]);
            transform.scale = toVec3(ji["localScale"]);
            const auto& q = ji["rotation"];
            transform.rotation = glm::quat(q[3].get<float>(), q[0].get<float>(),
                                           q[1].get<float>(), q[2].get<float>());
            InstanceData instanceData{
                .name = ji["name"],
                .meshName = ji.value("meshName", mBLASDatas[meshIndex].meshName),
                .transform = transform,
                .meshIndex = meshIndex,
                .materialIndex = ji["materialIndex"],
                .indexCount = mBLASDatas[meshIndex].indexCount
            };
            mInstances.push_back(instanceData);
        }
    }

    void SceneLoader::buildEmissiveAliasTables() {
        std::vector<bool> emissiveMesh(mBLASDatas.size(), false);
        for (const auto& instance : mInstances) {
            if (mMaterials[instance.materialIndex].luminance > 0.0f)
                emissiveMesh[instance.meshIndex] = true;
        }
        for (size_t i = 0; i < mBLASDatas.size(); ++i) {
            if (emissiveMesh[i]) buildAlias(mBLASDatas[i]);
        }
    }

    json SceneLoader::save() const {
        json scene = mJson;
        scene["version"] = 2;
        scene["directLight"]["direction"] = { mDirectLight.dir.x, mDirectLight.dir.y, mDirectLight.dir.z };
        scene["directLight"]["intensity"] = mDirectLight.intensity;

        if (mSkyboxIndex != UINT32_MAX) scene["skybox"] = mSkyboxPath;
        else scene.erase("skybox");

        json materials = json::array();
        for (const auto& material : mMaterials) {
            json jm;
            jm["name"]          = material.name;
            jm["color"]         = { material.baseColor.r, material.baseColor.g, material.baseColor.b };
            jm["emissionColor"] = { material.emissionColor.r, material.emissionColor.g, material.emissionColor.b };
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
            if (!material.baseColorTexName.empty()) jm["baseColorTex"] = material.baseColorTexName;
            if (!material.normalTexName.empty()) jm["normalTex"] = material.normalTexName;
            if (!material.metalRoughnessTexName.empty()) jm["metalRoughnessTex"] = material.metalRoughnessTexName;
            if (!material.clearcoatTexName.empty()) jm["clearcoatTex"] = material.clearcoatTexName;
            if (!material.clearcoatRoughnessTexName.empty()) jm["clearcoatRoughnessTex"] = material.clearcoatRoughnessTexName;
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
            ji["modelIndex"]    = mBLASDatas[instance.meshIndex].modelIndex;
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
}