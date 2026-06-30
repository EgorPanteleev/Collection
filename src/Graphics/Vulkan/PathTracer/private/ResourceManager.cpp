//
// Created by igor on 6/12/26.
//

#include "ResourceManager.hpp"
#include <algorithm>
#include <filesystem>

namespace crv::graphics::vulkan {
    static std::string relativeToAssets(const std::string& path) {
        std::error_code ec;
        const std::filesystem::path rel = std::filesystem::relative(path, ASSETS_PATH, ec);
        return (!ec && !rel.empty()) ? rel.generic_string() : path;
    }

    ResourceManager::ResourceManager(const ResourceManagerCreateInfo& info):
    mContext(info.context) {
        createLoader();
    }

    void ResourceManager::load(const json& json) {
        mSceneLoader.loadScene(json);
        buildTLAS();
        createBuffers();
    }

    void ResourceManager::updateInstanceTransform(const uint32_t index) {
        updateInstance(index);
        const InstanceData& instance = mSceneLoader.mInstances[index];
        InstanceData::AS asInstance =
            instance.vkAS(index, mSceneLoader.mBLASDatas[instance.meshIndex].blas.deviceAddress());
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
            .instanceCount = static_cast<uint32_t>(mSceneLoader.mInstances.size())
        };
        mTLAS.update(updateInfo);
        endCommandBuffer(cmdData, mContext->queue(QueueFamilyType::GRAPHICS));
    }

    void ResourceManager::updateInstance(const uint32_t index) {
        InstanceData::GPU instanceGPU = mSceneLoader.mInstances[index].gpu();
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

    std::vector<uint32_t> ResourceManager::duplicateInstances(const std::vector<uint32_t>& indices) {
        auto& instances = mSceneLoader.mInstances;
        std::vector<uint32_t> newIndices;
        newIndices.reserve(indices.size());
        for (const uint32_t index : indices) {
            if (index >= instances.size()) continue;
            const InstanceData copy = instances[index];
            instances.push_back(copy);
            newIndices.push_back(static_cast<uint32_t>(instances.size() - 1));
        }
        if (!newIndices.empty()) rebuildInstanceBuffers();
        return newIndices;
    }

    void ResourceManager::removeInstances(const std::vector<uint32_t>& indices) {
        auto& instances = mSceneLoader.mInstances;
        std::vector<uint32_t> sorted(indices);
        std::sort(sorted.begin(), sorted.end(), std::greater<>());
        sorted.erase(std::unique(sorted.begin(), sorted.end()), sorted.end());
        if (sorted.empty() || sorted.size() >= instances.size()) return;
        for (const uint32_t index : sorted) {
            if (index < instances.size()) instances.erase(instances.begin() + index);
        }
        rebuildInstanceBuffers();
    }

    uint32_t ResourceManager::addMaterial(const Material& material) {
        mSceneLoader.mMaterials.push_back(material);
        buildMaterialBuffer();
        return static_cast<uint32_t>(mSceneLoader.mMaterials.size() - 1);
    }

    void ResourceManager::buildMaterialBuffer() {
        SSBOData ssboData{};
        const auto materialsGPU = Material::gpu(mSceneLoader.mMaterials);
        ssboData.add(materialsGPU, mMaterialBuffer);
        ssboData.createAll(mContext, QueueFamilyType::GRAPHICS);
    }

    void ResourceManager::updateMaterial(const uint32_t index) {
        Material::GPU materialGPU = mSceneLoader.mMaterials[index].gpu();
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

    uint32_t ResourceManager::addBaseColorTexture(const std::string& path, const uint32_t materialIndex) {
        const cm::Texture cmTexture = cm::AbsLoader::loadTexture(path, cm::Texture::BASE_COLOR);
        mSceneLoader.mTextures.push_back(toTexture(mContext, cmTexture));
        const auto index = static_cast<uint32_t>(mSceneLoader.mTextures.size() - 1);
        Material& material = mSceneLoader.mMaterials[materialIndex];
        material.baseColorTexIndex = index;
        material.baseColorTexName = std::filesystem::path(path).filename().string();
        material.baseColorTexPath = relativeToAssets(path);
        updateMaterial(materialIndex);
        return index;
    }

    uint32_t ResourceManager::addNormalTexture(const std::string& path, const uint32_t materialIndex) {
        const cm::Texture cmTexture = cm::AbsLoader::loadTexture(path, cm::Texture::NORMAL);
        mSceneLoader.mTextures.push_back(toTexture(mContext, cmTexture));
        const auto index = static_cast<uint32_t>(mSceneLoader.mTextures.size() - 1);
        Material& material = mSceneLoader.mMaterials[materialIndex];
        material.normalTexIndex = index;
        material.normalTexName = std::filesystem::path(path).filename().string();
        material.normalTexPath = relativeToAssets(path);
        updateMaterial(materialIndex);
        return index;
    }

    uint32_t ResourceManager::addClearcoatTexture(const std::string& path, const uint32_t materialIndex) {
        const cm::Texture cmTexture = cm::AbsLoader::loadTexture(path, cm::Texture::CLEARCOAT);
        mSceneLoader.mTextures.push_back(toTexture(mContext, cmTexture));
        const auto index = static_cast<uint32_t>(mSceneLoader.mTextures.size() - 1);
        Material& material = mSceneLoader.mMaterials[materialIndex];
        material.clearcoatTexIndex = index;
        material.clearcoatTexName = std::filesystem::path(path).filename().string();
        material.clearcoatTexPath = relativeToAssets(path);
        updateMaterial(materialIndex);
        return index;
    }

    uint32_t ResourceManager::addClearcoatRoughnessTexture(const std::string& path, const uint32_t materialIndex) {
        const cm::Texture cmTexture = cm::AbsLoader::loadTexture(path, cm::Texture::CLEARCOAT_ROUGHNESS);
        mSceneLoader.mTextures.push_back(toTexture(mContext, cmTexture));
        const auto index = static_cast<uint32_t>(mSceneLoader.mTextures.size() - 1);
        Material& material = mSceneLoader.mMaterials[materialIndex];
        material.clearcoatRoughnessTexIndex = index;
        material.clearcoatRoughnessTexName = std::filesystem::path(path).filename().string();
        material.clearcoatRoughnessTexPath = relativeToAssets(path);
        updateMaterial(materialIndex);
        return index;
    }

    uint32_t ResourceManager::addSkybox(const std::string& path) {
        mSceneLoader.mTextures.push_back(toTexture(mContext, cm::AbsLoader::loadSkybox(path)));
        mSceneLoader.mSkyboxIndex = static_cast<uint32_t>(mSceneLoader.mTextures.size() - 1);
        mSceneLoader.mSkyboxName = std::filesystem::path(path).filename().string();
        std::error_code ec;
        const std::filesystem::path relative = std::filesystem::relative(path, ASSETS_PATH, ec);
        mSceneLoader.mSkyboxPath = (!ec && !relative.empty()) ? relative.generic_string() : path;
        return mSceneLoader.mSkyboxIndex;
    }

    void ResourceManager::removeSkybox() {
        mSceneLoader.mSkyboxIndex = UINT32_MAX;
        mSceneLoader.mSkyboxName.clear();
        mSceneLoader.mSkyboxPath.clear();
    }

    uint32_t ResourceManager::addMetalRoughnessTexture(const std::string& path, const uint32_t materialIndex) {
        const cm::Texture cmTexture = cm::AbsLoader::loadTexture(path, cm::Texture::METAL_ROUGHNESS);
        mSceneLoader.mTextures.push_back(toTexture(mContext, cmTexture));
        const auto index = static_cast<uint32_t>(mSceneLoader.mTextures.size() - 1);
        Material& material = mSceneLoader.mMaterials[materialIndex];
        material.metalRoughnessTexIndex = index;
        material.metalRoughnessTexName = std::filesystem::path(path).filename().string();
        material.metalRoughnessTexPath = relativeToAssets(path);
        updateMaterial(materialIndex);
        return index;
    }

    void ResourceManager::updateEmissiveIndices() {
        auto& indices   = mSceneLoader.mEmissiveIndices;
        const auto& instances = mSceneLoader.mInstances;
        const auto& materials = mSceneLoader.mMaterials;

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

    void ResourceManager::createLoader() {
        const SceneLoaderCreateInfo createInfo {
            .context = mContext
        };
        mSceneLoader = SceneLoader(createInfo);
    }

    void ResourceManager::buildTLAS() {
        const size_t instancesSize = sizeof(InstanceData::AS) * mSceneLoader.mInstances.size();
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
        asInstances.reserve(mSceneLoader.mInstances.size());
        for (size_t i = 0; i < mSceneLoader.mInstances.size(); ++i) {
            const InstanceData& instance = mSceneLoader.mInstances[i];
            const AccelerationStructure& blas = mSceneLoader.mBLASDatas[instance.meshIndex].blas;
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
            .instanceCount = static_cast<uint32_t>(mSceneLoader.mInstances.size())
        };
        mTLAS = AccelerationStructure(tlasCreateInfo);
        endCommandBuffer(cmdData, mContext->queue(QueueFamilyType::GRAPHICS));
    }

    void ResourceManager::createBuffers() {
        SSBOData ssboData{};
        const auto blasDatasGPU = BLASData::gpu(mContext->device(), mSceneLoader.mBLASDatas);
        ssboData.add(blasDatasGPU, mBLASBuffer);
        const auto instancesGPU = InstanceData::gpu(mSceneLoader.mInstances);
        ssboData.add(instancesGPU, mInstanceBuffer);
        const uint32_t emissiveCapacity = std::max<uint32_t>(mSceneLoader.mInstances.size(), 1u);
        std::vector emissiveIndices(emissiveCapacity, 0u);
        std::copy(mSceneLoader.mEmissiveIndices.begin(), mSceneLoader.mEmissiveIndices.end(), emissiveIndices.begin());
        ssboData.add(emissiveIndices, mEmissiveInstanceBuffer);
        const auto materialsGPU = Material::gpu(mSceneLoader.mMaterials);
        ssboData.add(materialsGPU, mMaterialBuffer);
        ssboData.createAll(mContext, QueueFamilyType::GRAPHICS);
    }

    void ResourceManager::rebuildInstanceBuffers() {
        buildTLAS();

        auto& emissive = mSceneLoader.mEmissiveIndices;
        emissive.clear();
        for (uint32_t i = 0; i < mSceneLoader.mInstances.size(); ++i) {
            if (mSceneLoader.mMaterials[mSceneLoader.mInstances[i].materialIndex].luminance == 0) continue;
            emissive.push_back(i);
        }

        SSBOData ssboData{};
        const auto instancesGPU = InstanceData::gpu(mSceneLoader.mInstances);
        ssboData.add(instancesGPU, mInstanceBuffer);
        const uint32_t emissiveCapacity = std::max<uint32_t>(mSceneLoader.mInstances.size(), 1u);
        std::vector emissiveIndices(emissiveCapacity, 0u);
        std::copy(emissive.begin(), emissive.end(), emissiveIndices.begin());
        ssboData.add(emissiveIndices, mEmissiveInstanceBuffer);
        ssboData.createAll(mContext, QueueFamilyType::GRAPHICS);
    }
}
