//
// Created by igor on 6/12/26.
//

#include "ResourceManager.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>

namespace crv::graphics::vulkan {
    namespace {
        constexpr double ENV_PI = 3.14159265358979323846;

        float halfToFloat(uint16_t h) {
            const uint32_t sign = (h >> 15) & 0x1u;
            uint32_t       exp  = (h >> 10) & 0x1Fu;
            uint32_t       mant = h & 0x3FFu;
            uint32_t       bits;
            if (exp == 0) {
                if (mant == 0) {
                    bits = sign << 31;
                } else {
                    exp = 127 - 15 + 1;
                    while ((mant & 0x400u) == 0) { mant <<= 1; --exp; }
                    mant &= 0x3FFu;
                    bits = (sign << 31) | (exp << 23) | (mant << 13);
                }
            } else if (exp == 0x1Fu) {
                bits = (sign << 31) | (0xFFu << 23) | (mant << 13);
            } else {
                bits = (sign << 31) | ((exp - 15 + 127) << 23) | (mant << 13);
            }
            float out;
            std::memcpy(&out, &bits, sizeof(out));
            return out;
        }

        struct EnvDistribution {
            uint32_t           width  = 0;
            uint32_t           height = 0;
            float              integral = 0.0f;
            std::vector<float> condFunc;
            std::vector<float> condCdf;
            std::vector<float> marginalCdf;
        };

        EnvDistribution buildEnvDistribution2D(const uint16_t* pixels, uint32_t width, uint32_t height) {
            EnvDistribution d;
            d.width  = width;
            d.height = height;
            const uint32_t nu = width;
            const uint32_t nv = height;
            d.condFunc.resize(static_cast<size_t>(nu) * nv);
            d.condCdf.resize(static_cast<size_t>(nu + 1) * nv);
            std::vector<float> marginalFunc(nv);

            auto sanitize = [](float v) {
                if (std::isnan(v)) return 0.0f;
                return std::clamp(v, 0.0f, 65504.0f);
            };
            for (uint32_t y = 0; y < nv; ++y) {
                const float sinTheta = static_cast<float>(std::sin(ENV_PI * (y + 0.5) / nv));
                float* func = &d.condFunc[static_cast<size_t>(y) * nu];
                for (uint32_t x = 0; x < nu; ++x) {
                    const uint16_t* p = pixels + (static_cast<size_t>(y) * nu + x) * 4;
                    const float lum = 0.2126f * sanitize(halfToFloat(p[0]))
                                    + 0.7152f * sanitize(halfToFloat(p[1]))
                                    + 0.0722f * sanitize(halfToFloat(p[2]));
                    func[x] = lum * sinTheta;
                }
                float* cdf = &d.condCdf[static_cast<size_t>(y) * (nu + 1)];
                cdf[0] = 0.0f;
                for (uint32_t x = 1; x <= nu; ++x) cdf[x] = cdf[x - 1] + func[x - 1] / nu;
                const float funcInt = cdf[nu];
                if (funcInt == 0.0f) {
                    for (uint32_t x = 1; x <= nu; ++x) cdf[x] = static_cast<float>(x) / nu;
                } else {
                    for (uint32_t x = 1; x <= nu; ++x) cdf[x] /= funcInt;
                }
                marginalFunc[y] = funcInt;
            }

            d.marginalCdf.resize(nv + 1);
            d.marginalCdf[0] = 0.0f;
            for (uint32_t y = 1; y <= nv; ++y)
                d.marginalCdf[y] = d.marginalCdf[y - 1] + marginalFunc[y - 1] / nv;
            const float marginalInt = d.marginalCdf[nv];
            if (marginalInt == 0.0f) {
                for (uint32_t y = 1; y <= nv; ++y) d.marginalCdf[y] = static_cast<float>(y) / nv;
            } else {
                for (uint32_t y = 1; y <= nv; ++y) d.marginalCdf[y] /= marginalInt;
            }
            d.integral = marginalInt;
            return d;
        }

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
    mContext(info.context), mScene(info.scene) {}

    void ResourceManager::load(const json& json) {
        mScene->load(json);
        mTextures.reserve(mScene->mTextureSources.size());
        for (const cm::Texture& source : mScene->mTextureSources)
            mTextures.push_back(toTexture(mContext, source));
        buildMeshes();
        if (mScene->mSkyboxIndex != UINT32_MAX && !mScene->mSkyboxPath.empty())
            buildEnvDistribution(cm::AbsLoader::loadSkybox(ASSETS_PATH + mScene->mSkyboxPath));
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
        buildEnvDistribution(skybox);
        mTextures.push_back(toTexture(mContext, skybox));
        return sourceIndex;
    }

    void ResourceManager::disableSkybox() {
        disableEnvDistribution();
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

    void ResourceManager::disableEnvDistribution() {
        mEnvMarginalCdfAddr = 0;
        mEnvCondCdfAddr     = 0;
        mEnvCondFuncAddr    = 0;
        mEnvIntegral        = 0.0f;
    }

    void ResourceManager::buildEnvDistribution(const cm::Texture& skybox) {
        disableEnvDistribution();
        if (skybox.mDataByLevel.empty() || skybox.mFormat != cm::Texture::R16G16B16A16_SFLOAT) return;
        const auto& level  = skybox.mDataByLevel[0];
        const auto* pixels = static_cast<const uint16_t*>(level.data);
        if (pixels == nullptr || level.width == 0 || level.height == 0) return;

        const EnvDistribution dist = buildEnvDistribution2D(pixels, level.width, level.height);
        if (dist.integral <= 0.0f) return;

        auto fillStorageBuffer = [this](Buffer& dst, const std::vector<float>& data) {
            const size_t size = sizeof(float) * data.size();
            const BufferCreateInfo createInfo{
                .allocator = mContext->allocator(),
                .size = size,
                .bufferUsage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                .memoryUsage = VMA_MEMORY_USAGE_GPU_ONLY
            };
            dst = Buffer(createInfo);
            const CopyDataToGPUBufferInfo copyInfo{
                .data = const_cast<float*>(data.data()),
                .size = size,
                .allocator = mContext->allocator(),
                .buffer = dst.get(),
                .device = mContext->device(),
                .queueFamilyIndex = mContext->familyIndex(QueueFamilyType::GRAPHICS).value(),
                .queue = mContext->queue(QueueFamilyType::GRAPHICS)
            };
            Buffer::copy(copyInfo);
        };

        fillStorageBuffer(mEnvMarginalCdfBuffer, dist.marginalCdf);
        fillStorageBuffer(mEnvCondCdfBuffer,     dist.condCdf);
        fillStorageBuffer(mEnvCondFuncBuffer,    dist.condFunc);
        mEnvMarginalCdfAddr = mEnvMarginalCdfBuffer.deviceAddress(mContext->device());
        mEnvCondCdfAddr     = mEnvCondCdfBuffer.deviceAddress(mContext->device());
        mEnvCondFuncAddr    = mEnvCondFuncBuffer.deviceAddress(mContext->device());
        mEnvIntegral        = dist.integral;
    }
}
