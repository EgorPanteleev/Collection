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
    }

    ResourceManager::ResourceManager(const ResourceManagerCreateInfo& info):
    mContext(info.context), mScene(info.scene), mEnvMap(info.context) { build(); }

    void ResourceManager::build() {
        mTextures.reserve(mScene->mTextureSources.size());
        for (const cm::Texture& source : mScene->mTextureSources)
            mTextures.push_back(toTexture(mContext, source));
        buildMeshes();
        if (mScene->mSkyboxIndex != UINT32_MAX && !mScene->mSkyboxPath.empty())
            mEnvMap.build(cm::AbsLoader::loadSkybox(ASSETS_PATH + mScene->mSkyboxPath));
        buildTLAS();
        createBuffers();
    }

    void ResourceManager::updateInstance(const uint32_t index) {
        updateInstanceData(index);
        const InstanceData& instance = mScene->mInstances[index];
        InstanceData::AS asInstance =
            instance.vkAS(index, mBLASDatas[instance.meshIndex].blas.deviceAddress());
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
        auto [commandBuffer, cmdData] = beginCommandBuffer(mContext, QueueFamilyType::GRAPHICS);
        const TLASUpdateInfo updateInfo {
            .commandBuffer = commandBuffer,
            .instanceCount = static_cast<uint32_t>(mScene->mInstances.size())
        };
        mTLAS.update(updateInfo);
        endCommandBuffer(cmdData, mContext->queue(QueueFamilyType::GRAPHICS));
    }

    void ResourceManager::updateInstanceData(const uint32_t index) {
        InstanceData::GPU instanceGPU = mScene->mInstances[index].gpu();
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
        updateEmissiveIndices();
    }

    void ResourceManager::rebuildInstances() {
        rebuildInstanceBuffers();
    }

    void ResourceManager::rebuildMaterials() {
        buildMaterialBuffer();
    }

    void ResourceManager::buildMaterialBuffer() {
        const auto materialsGPU = Material::gpu(mScene->mMaterials);
        SSBOBuilder(mContext, QueueFamilyType::GRAPHICS)
            .add(materialsGPU, mMaterialBuffer);
    }

    void ResourceManager::updateMaterial(const uint32_t index) {
        Material::GPU materialGPU = mScene->mMaterials[index].gpu();
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
        updateEmissiveIndices();
    }

    uint32_t ResourceManager::uploadTexture(const uint32_t sourceIndex) {
        mTextures.push_back(toTexture(mContext, mScene->mTextureSources[sourceIndex]));
        return sourceIndex;
    }

    uint32_t ResourceManager::uploadSkybox(const uint32_t sourceIndex) {
        const cm::Texture& skybox = mScene->mTextureSources[sourceIndex];
        mEnvMap.build(skybox);
        mTextures.push_back(toTexture(mContext, skybox));
        return sourceIndex;
    }

    void ResourceManager::disableSkybox() {
        mEnvMap.disable();
    }

    void ResourceManager::updateEmissiveIndices() {
        auto& indices   = mScene->mEmissiveIndices;
        const auto& instances = mScene->mInstances;
        const auto& materials = mScene->mMaterials;

        indices.clear();
        for (uint32_t i = 0; i < instances.size(); ++i) {
            if (materials[instances[i].materialIndex].luminance == 0) continue;
            indices.push_back(i);
        }
        if (indices.empty()) return;

        const CopyDataToGPUBufferInfo copyInfo {
            .data = indices.data(),
            .srcOffset = 0,
            .dstOffset = 0,
            .size = sizeof(uint32_t) * indices.size(),
            .allocator = mContext->allocator(),
            .buffer = mEmissiveInstanceBuffer.get(),
            .device = mContext->device(),
            .queueFamilyIndex = mContext->familyIndex(QueueFamilyType::GRAPHICS).value(),
            .queue = mContext->queue(QueueFamilyType::GRAPHICS)
        };
        Buffer::copy(copyInfo);
    }

    void ResourceManager::buildMeshes() {
        auto [commandBuffer, cmdData] = beginCommandBuffer(mContext->device(),
            mContext->familyIndex(QueueFamilyType::GRAPHICS).value());
        mBLASDatas.reserve(mScene->mMeshes.size());
        for (MeshData& mesh : mScene->mMeshes) {
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
        for (const auto& instance : mScene->mInstances) {
            if (mScene->mMaterials[instance.materialIndex].luminance > 0.0f)
                emissiveMesh[instance.meshIndex] = true;
        }
        for (size_t i = 0; i < mBLASDatas.size(); ++i) {
            if (emissiveMesh[i]) buildAlias(mBLASDatas[i], mScene->mMeshes[i].triAreas);
        }
    }

    void ResourceManager::buildTLAS() {
        const size_t instancesSize = sizeof(InstanceData::AS) * mScene->mInstances.size();
        const BufferCreateInfo instanceBufferCreateInfo {
            .allocator = mContext->allocator(),
            .size = instancesSize,
            .bufferUsage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                           VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .memoryUsage = VMA_MEMORY_USAGE_GPU_ONLY
        };
        mASInstanceBuffer = Buffer(instanceBufferCreateInfo);
        std::vector<InstanceData::AS> asInstances{};
        asInstances.reserve(mScene->mInstances.size());
        for (size_t i = 0; i < mScene->mInstances.size(); ++i) {
            const InstanceData& instance = mScene->mInstances[i];
            const AccelerationStructure& blas = mBLASDatas[instance.meshIndex].blas;
            asInstances.push_back(instance.vkAS(i, blas.deviceAddress()));
        }
        const CopyDataToGPUBufferInfo instanceCopyInfo {
            .data = asInstances.data(),
            .size = instancesSize,
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
            .instanceCount = static_cast<uint32_t>(mScene->mInstances.size())
        };
        mTLAS = AccelerationStructure(tlasCreateInfo);
        endCommandBuffer(cmdData, mContext->queue(QueueFamilyType::GRAPHICS));
    }

    void ResourceManager::createBuffers() {
        const auto blasDatasGPU = BLASData::gpu(mContext->device(), mBLASDatas);
        const auto instancesGPU = InstanceData::gpu(mScene->mInstances);
        const uint32_t emissiveCapacity = std::max<uint32_t>(mScene->mInstances.size(), 1u);
        std::vector emissiveIndices(emissiveCapacity, 0u);
        std::copy(mScene->mEmissiveIndices.begin(), mScene->mEmissiveIndices.end(), emissiveIndices.begin());
        const auto materialsGPU = Material::gpu(mScene->mMaterials);
        SSBOBuilder(mContext, QueueFamilyType::GRAPHICS)
            .add(blasDatasGPU   , mBLASBuffer            )
            .add(instancesGPU   , mInstanceBuffer        )
            .add(emissiveIndices, mEmissiveInstanceBuffer)
            .add(materialsGPU   , mMaterialBuffer        );
    }

    void ResourceManager::rebuildInstanceBuffers() {
        buildTLAS();

        auto& emissive = mScene->mEmissiveIndices;
        emissive.clear();
        for (uint32_t i = 0; i < mScene->mInstances.size(); ++i) {
            if (mScene->mMaterials[mScene->mInstances[i].materialIndex].luminance == 0) continue;
            emissive.push_back(i);
        }

        const auto instancesGPU = InstanceData::gpu(mScene->mInstances);
        const uint32_t emissiveCapacity = std::max<uint32_t>(mScene->mInstances.size(), 1u);
        std::vector emissiveIndices(emissiveCapacity, 0u);
        std::copy(emissive.begin(), emissive.end(), emissiveIndices.begin());
        SSBOBuilder(mContext, QueueFamilyType::GRAPHICS)
            .add(instancesGPU   , mInstanceBuffer        )
            .add(emissiveIndices, mEmissiveInstanceBuffer);
    }

}
