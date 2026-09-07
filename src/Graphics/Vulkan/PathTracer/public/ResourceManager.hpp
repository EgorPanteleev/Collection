//
// Created by igor on 6/12/26.
//

#ifndef COLLECTION_RESOURCEMANAGER_HPP
#define COLLECTION_RESOURCEMANAGER_HPP

#include "Context.hpp"
#include "Model/Scene.hpp"

namespace crv::graphics::vulkan {
    struct ResourceManagerCreateInfo {
        Context* context = nullptr;
        Scene*   scene   = nullptr;
    };

    class ResourceManager {
    public:
        ResourceManager() = default;
        explicit ResourceManager(const ResourceManagerCreateInfo& info);
        void load(const json& json);
        void updateInstance(uint32_t index);
        void updateInstanceData(uint32_t index);
        void updateMaterial(uint32_t index);
        void updateEmissiveIndices();
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
        [[nodiscard]] float envIntegral() const { return mEnvIntegral; }
        [[nodiscard]] uint64_t envMarginalCdfAddr() const { return mEnvMarginalCdfAddr; }
        [[nodiscard]] uint64_t envCondCdfAddr() const { return mEnvCondCdfAddr; }
        [[nodiscard]] uint64_t envCondFuncAddr() const { return mEnvCondFuncAddr; }
    private:
        void buildMeshes();
        void buildAlias(BLASData& blasData, const std::vector<float>& triAreas);
        void buildEmissiveAliasTables();
        void buildTLAS();
        void createBuffers();
        void rebuildInstanceBuffers();
        void buildMaterialBuffer();
        void buildEnvDistribution(const cm::Texture& skybox);
        void disableEnvDistribution();

        Context*              mContext          = nullptr;
        Scene*                mScene            = nullptr;
        std::vector<BLASData> mBLASDatas{};
        Buffer                mBLASBuffer             = CRV_NULL_HANDLE;
        Buffer                mASInstanceBuffer       = CRV_NULL_HANDLE;
        Buffer                mInstanceBuffer         = CRV_NULL_HANDLE;
        Buffer                mEmissiveInstanceBuffer = CRV_NULL_HANDLE;
        Buffer                mMaterialBuffer         = CRV_NULL_HANDLE;
        AccelerationStructure mTLAS                   = CRV_NULL_HANDLE;
        std::vector<Texture>  mTextures{};

        Buffer   mEnvMarginalCdfBuffer = CRV_NULL_HANDLE;
        Buffer   mEnvCondCdfBuffer     = CRV_NULL_HANDLE;
        Buffer   mEnvCondFuncBuffer    = CRV_NULL_HANDLE;
        uint64_t mEnvMarginalCdfAddr   = 0;
        uint64_t mEnvCondCdfAddr       = 0;
        uint64_t mEnvCondFuncAddr      = 0;
        float    mEnvIntegral          = 0.0f;
    };
}

#endif //COLLECTION_RESOURCEMANAGER_HPP