//
// Created by igor on 6/12/26.
//

#include "View/ResourceManager.hpp"
#include <algorithm>

namespace crv::graphics::vulkan {
    namespace {
        std::vector<AliasEntry> buildAliasTable(const std::vector<float>& weights) {
            const size_t n = weights.size();
            std::vector<AliasEntry> table(n);
            double sum = 0.0;
            for (const float w : weights) sum += w;

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
            for (const uint32_t l : large) { table[l].prob = 1.0f; table[l].alias = l; }
            for (const uint32_t s : small) { table[s].prob = 1.0f; table[s].alias = s; }
            return table;
        }

        float emissiveWorldArea(const glm::mat4& world, const MeshData& mesh) {
            const glm::mat3 m = glm::mat3(world);
            double area = 0.0;
            for (size_t t = 0; t + 3 <= mesh.indices.size(); t += 3) {
                const glm::vec3 p0 = mesh.vertices[mesh.indices[t + 0]].pos;
                const glm::vec3 p1 = mesh.vertices[mesh.indices[t + 1]].pos;
                const glm::vec3 p2 = mesh.vertices[mesh.indices[t + 2]].pos;
                area += 0.5 * glm::length(glm::cross(m * (p1 - p0), m * (p2 - p0)));
            }
            return static_cast<float>(area);
        }
    }

    ResourceManager::ResourceManager(const ResourceManagerCreateInfo& info):
    mContext(info.context), mScene(info.scene), mEnvMap(info.context) { build(); }

    void ResourceManager::build() {
        syncTextures();
        buildMeshes();
        const uint32_t skyboxIndex = mScene->skyboxIndex();
        if (skyboxIndex < mScene->textureSources().size())
            mEnvMap.build(mScene->textureSources()[skyboxIndex]);
        buildTLAS();
        createBuffers();
    }

    void ResourceManager::addModel() {
        syncTextures();
        buildMeshes();
        buildBLASBuffer();
        rebuildInstanceBuffers();
        buildMaterialBuffer();
    }

    void ResourceManager::uploadInstanceAS(const uint32_t index) {
        const InstanceData& instance = mScene->instances()[index];
        InstanceData::AS asInstance =
            instance.vkAS(index, mBLASDatas[instance.meshIndex].blas.deviceAddress());
        if (alphaMasked(instance)) asInstance.flags |= VK_GEOMETRY_INSTANCE_FORCE_NO_OPAQUE_BIT_KHR;
        const CopyDataToGPUBufferInfo copyInfo {
            .data = &asInstance,
            .srcOffset = 0,
            .dstOffset = sizeof(InstanceData::AS) * index,
            .size = sizeof(InstanceData::AS),
            .allocator = mContext->allocator(),
            .buffer = mASInstanceBuffer.get(),
            .device = mContext->device(),
            .queueFamilyIndex = mContext->familyIndex(QueueFamilyType::GRAPHICS).value(),
            .queue = mContext->queue(QueueFamilyType::GRAPHICS)
        };
        Buffer::copy(copyInfo);
    }

    void ResourceManager::refreshTLAS() {
        auto [commandBuffer, cmdData] = beginCommandBuffer(mContext, QueueFamilyType::GRAPHICS);
        const TLASUpdateInfo updateInfo {
            .commandBuffer = commandBuffer,
            .instanceCount = static_cast<uint32_t>(mScene->instances().size())
        };
        mTLAS.update(updateInfo);
        endCommandBuffer(cmdData, mContext->queue(QueueFamilyType::GRAPHICS));
    }

    void ResourceManager::updateInstance(const uint32_t index) {
        if (mScene->instances()[index].isGroup()) return;
        uploadInstanceData(index);
        uploadInstanceAS(index);
    }

    bool ResourceManager::alphaMasked(const InstanceData& instance) const {
        if (instance.isGroup() || instance.materialIndex >= mScene->materials().size()) return false;
        return mScene->materials()[instance.materialIndex].opacity < 1.0f;
    }

    InstanceData::GPU ResourceManager::gpuInstance(const uint32_t index) const {
        const InstanceData& instance = mScene->instances()[index];
        InstanceData::GPU gpu = instance.gpu();
        if (instance.isGroup() || instance.materialIndex >= mScene->materials().size()) return gpu;
        if (mScene->materials()[instance.materialIndex].luminance > 0.0f)
            gpu.lightArea = emissiveWorldArea(instance.world, mScene->meshes()[instance.meshIndex]);
        return gpu;
    }

    std::vector<InstanceData::GPU> ResourceManager::gpuInstances() const {
        const auto count = static_cast<uint32_t>(mScene->instances().size());
        std::vector<InstanceData::GPU> res{};
        res.reserve(count);
        for (uint32_t i = 0; i < count; ++i) res.push_back(gpuInstance(i));
        return res;
    }

    void ResourceManager::uploadInstanceData(const uint32_t index) {
        InstanceData::GPU instanceGPU = gpuInstance(index);
        const CopyDataToGPUBufferInfo copyInfo {
            .data = &instanceGPU,
            .srcOffset = 0,
            .dstOffset = sizeof(InstanceData::GPU) * index,
            .size = sizeof(InstanceData::GPU),
            .allocator = mContext->allocator(),
            .buffer = mInstanceBuffer.get(),
            .device = mContext->device(),
            .queueFamilyIndex = mContext->familyIndex(QueueFamilyType::GRAPHICS).value(),
            .queue = mContext->queue(QueueFamilyType::GRAPHICS)
        };
        Buffer::copy(copyInfo);
    }

    void ResourceManager::refreshMaterialInstances(const uint32_t materialIndex) {
        if (materialIndex >= mScene->materials().size()) return;
        const Material& material = mScene->materials()[materialIndex];
        if (material.luminance <= 0.0f && material.opacity >= 1.0f) return;
        const auto& instances = mScene->instances();
        const auto count = static_cast<uint32_t>(instances.size());
        bool touched = false;
        for (uint32_t i = 0; i < count; ++i) {
            if (instances[i].isGroup() || instances[i].materialIndex != materialIndex) continue;
            updateInstance(i);
            touched = true;
        }
        if (touched) refreshTLAS();
    }

    void ResourceManager::updateInstanceData(const uint32_t index) {
        uploadInstanceData(index);
    }

    void ResourceManager::rebuildInstances() {
        rebuildInstanceBuffers();
    }

    void ResourceManager::rebuildMaterials() {
        buildMaterialBuffer();
    }

    void ResourceManager::buildBLASBuffer() {
        const auto blasDatasGPU = BLASData::gpu(mContext->device(), mBLASDatas);
        SSBOBuilder(mContext, QueueFamilyType::GRAPHICS)
            .add(blasDatasGPU, mBLASBuffer);
    }

    void ResourceManager::buildMaterialBuffer() {
        const auto materialsGPU = Material::gpu(mScene->materials());
        SSBOBuilder(mContext, QueueFamilyType::GRAPHICS)
            .add(materialsGPU, mMaterialBuffer);
    }

    void ResourceManager::updateMaterial(const uint32_t index) {
        if (index >= mScene->materials().size()) return;
        Material::GPU materialGPU = mScene->materials()[index].gpu();
        const CopyDataToGPUBufferInfo copyInfo {
            .data = &materialGPU,
            .srcOffset = 0,
            .dstOffset = sizeof(Material::GPU) * index,
            .size = sizeof(Material::GPU),
            .allocator = mContext->allocator(),
            .buffer = mMaterialBuffer.get(),
            .device = mContext->device(),
            .queueFamilyIndex = mContext->familyIndex(QueueFamilyType::GRAPHICS).value(),
            .queue = mContext->queue(QueueFamilyType::GRAPHICS)
        };
        Buffer::copy(copyInfo);
        refreshMaterialInstances(index);
        updateEmissiveIndices();
    }

    void ResourceManager::syncTextures() {
        const auto& sources = mScene->textureSources();
        mTextures.reserve(sources.size());
        for (size_t i = mTextures.size(); i < sources.size(); ++i)
            mTextures.push_back(toTexture(mContext, sources[i]));
    }

    uint32_t ResourceManager::uploadTexture(const uint32_t sourceIndex) {
        syncTextures();
        return sourceIndex;
    }

    uint32_t ResourceManager::uploadSkybox(const uint32_t sourceIndex) {
        if (sourceIndex >= mScene->textureSources().size()) return sourceIndex;
        mEnvMap.build(mScene->textureSources()[sourceIndex]);
        syncTextures();
        return sourceIndex;
    }

    void ResourceManager::disableSkybox() {
        mEnvMap.disable();
    }

    void ResourceManager::updateEmissiveIndices() {
        const auto lights = buildEmissiveLights();
        const CopyDataToGPUBufferInfo copyInfo {
            .data = lights.data(),
            .srcOffset = 0,
            .dstOffset = 0,
            .size = sizeof(EmissiveGPU) * lights.size(),
            .allocator = mContext->allocator(),
            .buffer = mEmissiveInstanceBuffer.get(),
            .device = mContext->device(),
            .queueFamilyIndex = mContext->familyIndex(QueueFamilyType::GRAPHICS).value(),
            .queue = mContext->queue(QueueFamilyType::GRAPHICS)
        };
        Buffer::copy(copyInfo);
    }

    void ResourceManager::buildMeshes() {
        const size_t start = mBLASDatas.size();
        const size_t total = mScene->meshes().size();
        if (start >= total) { buildEmissiveAliasTables(); return; }

        auto [commandBuffer, cmdData] = beginCommandBuffer(mContext->device(),
            mContext->familyIndex(QueueFamilyType::GRAPHICS).value());
        mBLASDatas.reserve(total);
        for (size_t meshIndex = start; meshIndex < total; ++meshIndex) {
            const MeshData& mesh = mScene->meshes()[meshIndex];
            mBLASDatas.emplace_back();
            BLASData& blasData = mBLASDatas.back();
            blasData.area = mesh.area;
            blasData.indexCount = mesh.indexCount;
            blasData.modelIndex = mesh.modelIndex;
            blasData.meshName = mesh.meshName;
            blasData.bbox = mesh.bbox;

            const size_t verticesSize = sizeof(Vertex) * mesh.vertices.size();
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
                .data = mesh.vertices.data(),
                .size = verticesSize,
                .allocator = mContext->allocator(),
                .buffer = blasData.vertexBuffer.get(),
                .device = mContext->device(),
                .queueFamilyIndex = mContext->familyIndex(QueueFamilyType::GRAPHICS).value(),
                .queue = mContext->queue(QueueFamilyType::GRAPHICS)
            };
            Buffer::copy(vertexCopyInfo);

            const size_t indicesSize = sizeof(uint32_t) * mesh.indices.size();
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
                .data = mesh.indices.data(),
                .size = indicesSize,
                .allocator = mContext->allocator(),
                .buffer = blasData.indexBuffer.get(),
                .device = mContext->device(),
                .queueFamilyIndex = mContext->familyIndex(QueueFamilyType::GRAPHICS).value(),
                .queue = mContext->queue(QueueFamilyType::GRAPHICS)
            };
            Buffer::copy(indexCopyInfo);

            const BLASCreateInfo blasCreateInfo{
                .commandBuffer = commandBuffer,
                .device = mContext->device(),
                .physicalDevice = mContext->physicalDevice(),
                .allocator = mContext->allocator(),
                .vertexAddress = blasData.vertexBuffer.deviceAddress(mContext->device()),
                .vertexStride = sizeof(Vertex),
                .vertexCount = static_cast<uint32_t>(mesh.vertices.size()),
                .indexAddress = blasData.indexBuffer.deviceAddress(mContext->device()),
                .indexCount = static_cast<uint32_t>(mesh.indices.size())
            };
            blasData.blas = AccelerationStructure(blasCreateInfo);
        }
        endCommandBuffer(cmdData, mContext->queue(QueueFamilyType::GRAPHICS));
        buildEmissiveAliasTables();
    }

    void ResourceManager::buildAlias(BLASData& blasData, const std::vector<float>& triAreas) {
        if (triAreas.empty() || blasData.aliasBuffer.get() != VK_NULL_HANDLE) return;
        auto aliasTable = buildAliasTable(triAreas);
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

    void ResourceManager::buildEmissiveAliasTables() {
        std::vector<bool> emissiveMesh(mBLASDatas.size(), false);
        for (const auto& instance : mScene->instances()) {
            if (instance.isGroup() || instance.meshIndex >= emissiveMesh.size()) continue;
            if (instance.materialIndex >= mScene->materials().size()) continue;
            if (mScene->materials()[instance.materialIndex].luminance > 0.0f)
                emissiveMesh[instance.meshIndex] = true;
        }
        for (size_t i = 0; i < mBLASDatas.size(); ++i) {
            if (emissiveMesh[i]) buildAlias(mBLASDatas[i], mScene->meshes()[i].triAreas);
        }
    }

    std::vector<EmissiveGPU> ResourceManager::buildEmissiveLights() {
        mScene->recomputeEmissiveIndices();
        const auto& indices  = mScene->emissiveIndices();
        const uint32_t capacity = std::max<uint32_t>(mScene->instances().size(), 1u);

        std::vector<float> weights(indices.size());
        for (size_t i = 0; i < indices.size(); ++i) {
            const InstanceData& instance = mScene->instances()[indices[i]];
            const float luminance = mScene->materials()[instance.materialIndex].luminance;
            const float area      = emissiveWorldArea(instance.world, mScene->meshes()[instance.meshIndex]);
            weights[i] = std::max(luminance * area, 0.0f);
        }
        double total = 0.0;
        for (const float w : weights) total += w;
        mEmissivePowerInv = total > 0.0 ? static_cast<float>(1.0 / total) : 0.0f;

        const std::vector<AliasEntry> alias = weights.empty() ? std::vector<AliasEntry>{} : buildAliasTable(weights);

        std::vector<EmissiveGPU> lights(capacity, EmissiveGPU{0u, 0u, 0.0f});
        for (size_t i = 0; i < indices.size(); ++i) {
            lights[i].instanceIndex = indices[i];
            lights[i].aliasIndex    = alias[i].alias;
            lights[i].aliasProb     = alias[i].prob;
        }
        return lights;
    }

    void ResourceManager::buildTLAS() {
        const size_t instancesSize = sizeof(InstanceData::AS) *
            std::max<size_t>(mScene->instances().size(), 1);
        const BufferCreateInfo instanceBufferCreateInfo {
            .allocator = mContext->allocator(),
            .size = instancesSize,
            .bufferUsage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                           VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .memoryUsage = VMA_MEMORY_USAGE_GPU_ONLY
        };
        mASInstanceBuffer = Buffer(instanceBufferCreateInfo);
        const VkDeviceAddress fallbackBlas = mBLASDatas.empty() ? 0 : mBLASDatas[0].blas.deviceAddress();
        std::vector<InstanceData::AS> asInstances{};
        asInstances.reserve(mScene->instances().size());
        for (size_t i = 0; i < mScene->instances().size(); ++i) {
            const InstanceData& instance = mScene->instances()[i];
            const VkDeviceAddress blas = instance.isGroup()
                ? fallbackBlas : mBLASDatas[instance.meshIndex].blas.deviceAddress();
            InstanceData::AS as = instance.vkAS(static_cast<uint32_t>(i), blas);
            if (instance.isGroup()) as.mask = 0;
            if (alphaMasked(instance)) as.flags |= VK_GEOMETRY_INSTANCE_FORCE_NO_OPAQUE_BIT_KHR;
            asInstances.push_back(as);
        }
        const CopyDataToGPUBufferInfo instanceCopyInfo {
            .data = asInstances.data(),
            .size = sizeof(InstanceData::AS) * asInstances.size(),
            .allocator = mContext->allocator(),
            .buffer = mASInstanceBuffer.get(),
            .device = mContext->device(),
            .queueFamilyIndex = mContext->familyIndex(QueueFamilyType::GRAPHICS).value(),
            .queue = mContext->queue(QueueFamilyType::GRAPHICS)
        };
        Buffer::copy(instanceCopyInfo);

        auto [commandBuffer, cmdData] = beginCommandBuffer(mContext->device(),
            mContext->familyIndex(QueueFamilyType::GRAPHICS).value());
        const TLASCreateInfo tlasCreateInfo {
            .commandBuffer = commandBuffer,
            .device = mContext->device(),
            .physicalDevice = mContext->physicalDevice(),
            .allocator = mContext->allocator(),
            .instanceAddress = mASInstanceBuffer.deviceAddress(mContext->device()),
            .instanceCount = static_cast<uint32_t>(mScene->instances().size())
        };
        mTLAS = AccelerationStructure(tlasCreateInfo);
        endCommandBuffer(cmdData, mContext->queue(QueueFamilyType::GRAPHICS));
    }

    void ResourceManager::createBuffers() {
        const auto blasDatasGPU   = BLASData::gpu(mContext->device(), mBLASDatas);
        const auto emissiveLights = buildEmissiveLights();
        const auto instancesGPU   = gpuInstances();
        const auto materialsGPU   = Material::gpu(mScene->materials());
        SSBOBuilder(mContext, QueueFamilyType::GRAPHICS)
            .add(blasDatasGPU   , mBLASBuffer            )
            .add(instancesGPU   , mInstanceBuffer        )
            .add(emissiveLights , mEmissiveInstanceBuffer)
            .add(materialsGPU   , mMaterialBuffer        );
    }

    void ResourceManager::rebuildInstanceBuffers() {
        buildTLAS();
        const auto emissiveLights = buildEmissiveLights();
        const auto instancesGPU   = gpuInstances();
        SSBOBuilder(mContext, QueueFamilyType::GRAPHICS)
            .add(instancesGPU   , mInstanceBuffer        )
            .add(emissiveLights , mEmissiveInstanceBuffer);
    }

}
