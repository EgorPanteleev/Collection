//
// Created by igor on 6/12/26.
//

#include "ResourceManager.hpp"

namespace crv::graphics::vulkan {
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
        auto emissiveIndices = mSceneLoader.mEmissiveIndices;
        if (emissiveIndices.empty()) emissiveIndices.push_back(UINT32_MAX);
        ssboData.add(emissiveIndices, mEmissiveInstanceBuffer);
        const auto materialsGPU = Material::gpu(mSceneLoader.mMaterials);
        ssboData.add(materialsGPU, mMaterialBuffer);
        ssboData.createAll(mContext, QueueFamilyType::GRAPHICS);
    }
}
