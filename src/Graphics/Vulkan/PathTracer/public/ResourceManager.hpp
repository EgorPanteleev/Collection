//
// Created by igor on 6/12/26.
//

#ifndef COLLECTION_RESOURCEMANAGER_HPP
#define COLLECTION_RESOURCEMANAGER_HPP

#include "Context.hpp"
#include "SceneLoader.hpp"

namespace crv::graphics::vulkan {
    struct ResourceManagerCreateInfo {
        Context* context = nullptr;
    };

    class ResourceManager {
    public:
        ResourceManager() = default;
        explicit ResourceManager(const ResourceManagerCreateInfo& info);
        void load(const json& json);
        [[nodiscard]] json saveScene() const { return mSceneLoader.save(); }
        void updateInstanceTransform(uint32_t index);
        void updateInstance(uint32_t index);
        void updateMaterial(uint32_t index);
        void updateEmissiveIndices();
        uint32_t addBaseColorTexture(const std::string& path, uint32_t materialIndex);
        uint32_t addNormalTexture(const std::string& path, uint32_t materialIndex);
        uint32_t addMetalRoughnessTexture(const std::string& path, uint32_t materialIndex);
        uint32_t addSkybox(const std::string& path);
        void removeSkybox();

        [[nodiscard]] DirectLight& directLight() { return mSceneLoader.mDirectLight; }
        [[nodiscard]] std::vector<InstanceData>& instances() { return mSceneLoader.mInstances; }
        [[nodiscard]] std::vector<uint32_t>& emissiveIndices() { return mSceneLoader.mEmissiveIndices; }
        [[nodiscard]] std::vector<BLASData>& blasDatas() { return mSceneLoader.mBLASDatas; }
        [[nodiscard]] std::vector<Material>& materials() { return mSceneLoader.mMaterials; }

        [[nodiscard]] AccelerationStructure& tlas() { return mTLAS; }
        [[nodiscard]] Buffer& blasBuffer() { return mBLASBuffer; }
        [[nodiscard]] Buffer& asInstanceBuffer() { return mASInstanceBuffer; }
        [[nodiscard]] Buffer& instanceBuffer() { return mInstanceBuffer; }
        [[nodiscard]] Buffer& emissiveInstanceBuffer() { return mEmissiveInstanceBuffer; }
        [[nodiscard]] Buffer& materialBuffer() { return mMaterialBuffer; }
        [[nodiscard]] std::vector<Texture>& textures() { return mSceneLoader.mTextures; }
        [[nodiscard]] uint32_t skyboxIndex() const { return mSceneLoader.mSkyboxIndex; }
        [[nodiscard]] const std::string& skyboxName() const { return mSceneLoader.mSkyboxName; }
    private:
        void createLoader();
        void buildTLAS();
        void createBuffers();

        Context*              mContext          = nullptr;
        SceneLoader           mSceneLoader{};
        Buffer                mBLASBuffer             = CRV_NULL_HANDLE;
        Buffer                mASInstanceBuffer       = CRV_NULL_HANDLE;
        Buffer                mInstanceBuffer         = CRV_NULL_HANDLE;
        Buffer                mEmissiveInstanceBuffer = CRV_NULL_HANDLE;
        Buffer                mMaterialBuffer         = CRV_NULL_HANDLE;
        AccelerationStructure mTLAS                   = CRV_NULL_HANDLE;
    };
}

#endif //COLLECTION_RESOURCEMANAGER_HPP