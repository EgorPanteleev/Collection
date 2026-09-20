//
// Created by igor on 6/12/26.
//

#ifndef COLLECTION_RESOURCEMANAGER_HPP
#define COLLECTION_RESOURCEMANAGER_HPP

#include "Context.hpp"
#include "Model/Scene.hpp"
#include "View/Types.hpp"
#include "View/EnvironmentMap.hpp"

namespace crv::graphics::vulkan {
    struct ResourceManagerCreateInfo {
        Context* context = nullptr;
        Scene*   scene   = nullptr;
    };

    class ResourceManager {
    public:
        ResourceManager() = default;
        explicit ResourceManager(const ResourceManagerCreateInfo& info);
        void build();
        void addModel();
        void updateInstance(uint32_t index);
        void updateInstanceData(uint32_t index);
        void updateMaterial(uint32_t index);
        void updateEmissiveIndices();
        void refreshTLAS();
        void rebuildInstances();
        void rebuildMaterials();
        uint32_t uploadTexture(uint32_t sourceIndex);
        uint32_t uploadSkybox(uint32_t sourceIndex);
        void disableSkybox();

        [[nodiscard]] std::vector<BLASData>& blasDatas() { return mBLASDatas; }
        [[nodiscard]] AccelerationStructure& tlas() { return mTLAS; }
        [[nodiscard]] Buffer& blasBuffer() { return mBLASBuffer; }
        [[nodiscard]] Buffer& asInstanceBuffer() { return mASInstanceBuffer; }
        [[nodiscard]] Buffer& instanceBuffer() { return mInstanceBuffer; }
        [[nodiscard]] Buffer& emissiveInstanceBuffer() { return mEmissiveInstanceBuffer; }
        [[nodiscard]] Buffer& materialBuffer() { return mMaterialBuffer; }
        [[nodiscard]] std::vector<Texture>& textures() { return mTextures; }
        [[nodiscard]] float envIntegral() const { return mEnvMap.integral(); }
        [[nodiscard]] float emissivePowerInv() const { return mEmissivePowerInv; }
        [[nodiscard]] uint64_t envMarginalCdfAddr() const { return mEnvMap.marginalCdfAddr(); }
        [[nodiscard]] uint64_t envCondCdfAddr() const { return mEnvMap.condCdfAddr(); }
        [[nodiscard]] uint64_t envCondFuncAddr() const { return mEnvMap.condFuncAddr(); }
    private:
        void buildMeshes();
        void buildAlias(BLASData& blasData, const std::vector<float>& triAreas);
        void buildEmissiveAliasTables();
        std::vector<EmissiveGPU> buildEmissiveLights();
        void buildTLAS();
        void createBuffers();
        void rebuildInstanceBuffers();
        void buildBLASBuffer();
        void buildMaterialBuffer();
        void syncTextures();
        [[nodiscard]] bool alphaMasked(const InstanceData& instance) const;
        [[nodiscard]] InstanceData::GPU gpuInstance(uint32_t index) const;
        [[nodiscard]] std::vector<InstanceData::GPU> gpuInstances() const;
        void uploadInstanceData(uint32_t index);
        void uploadInstanceAS(uint32_t index);
        void refreshMaterialInstances(uint32_t materialIndex);

        Context*              mContext          = nullptr;
        Scene*                mScene            = nullptr;
        std::vector<BLASData> mBLASDatas{};
        Buffer                mBLASBuffer             = CRV_NULL_HANDLE;
        Buffer                mASInstanceBuffer       = CRV_NULL_HANDLE;
        Buffer                mInstanceBuffer         = CRV_NULL_HANDLE;
        Buffer                mEmissiveInstanceBuffer = CRV_NULL_HANDLE;
        float                 mEmissivePowerInv       = 0.0f;
        Buffer                mMaterialBuffer         = CRV_NULL_HANDLE;
        AccelerationStructure mTLAS                   = CRV_NULL_HANDLE;
        std::vector<Texture>  mTextures{};
        EnvironmentMap        mEnvMap{};
    };
}

#endif //COLLECTION_RESOURCEMANAGER_HPP