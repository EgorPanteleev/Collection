//
// Created by igor on 6/10/26.
//

#ifndef COLLECTION_SCENELOADER_HPP
#define COLLECTION_SCENELOADER_HPP

#include <nlohmann/json.hpp>
using json = nlohmann::json;

#include "Loader.hpp"
#include "Message.hpp"
#include "Timer.hpp"
#include "CoreUtils.hpp"
#include "Types.hpp"

namespace crv::graphics::vulkan {
    namespace cu = utils;
    namespace cm = model;
    namespace fs = std::filesystem;

    inline glm::vec3 toVec3(const nlohmann::json& json) {
        return {
            json[0].get<float>(),
            json[1].get<float>(),
            json[2].get<float>()
        };
    }

    struct SceneLoaderCreateInfo {
        Context* context = nullptr;
    };

    class SceneLoader {
    public:
        SceneLoader() = default;
        explicit SceneLoader(const SceneLoaderCreateInfo& info);
        void loadScene(const json& scene);
        [[nodiscard]] json save() const;
        void buildEnvDistribution(const cm::Texture& skybox);
        void disableEnvDistribution();
    private:
        void loadModel(uint32_t modelIndex, const std::string& path);
        void loadMaterials();
        void buildAlias(BLASData& blasData);
        void applyResolvedMaterials();
        void loadResolvedTexture(const json& jm, const char* key, int textureType,
                                 uint32_t& texIndex, std::string& texName, std::string& texPath);
        void loadExplicitInstances();
        void buildEmissiveAliasTables();

        json     mJson{};
        Context* mContext  = nullptr;
        bool     mExplicit = false;
    public:
        DirectLight                  mDirectLight{};
        std::vector<BLASData>        mBLASDatas{};
        std::vector<InstanceData>    mInstances{};
        std::vector<uint32_t>        mEmissiveIndices{};
        std::vector<Material>        mMaterials{};
        std::vector<Texture>         mTextures{};
        uint32_t                     mSkyboxIndex = UINT32_MAX;
        std::string                  mSkyboxName{};
        std::string                  mSkyboxPath{};

        Buffer   mEnvMarginalCdfBuffer = CRV_NULL_HANDLE;
        Buffer   mEnvCondCdfBuffer     = CRV_NULL_HANDLE;
        Buffer   mEnvCondFuncBuffer    = CRV_NULL_HANDLE;
        uint64_t mEnvMarginalCdfAddr   = 0;
        uint64_t mEnvCondCdfAddr       = 0;
        uint64_t mEnvCondFuncAddr      = 0;
        float    mEnvIntegral          = 0.0f;
    };
}

#endif //COLLECTION_SCENELOADER_HPP