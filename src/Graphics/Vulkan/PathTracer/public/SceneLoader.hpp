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
    private:
        void loadModel(uint32_t modelIndex, const std::string& path);
        void loadMaterials();
        void buildAlias(BLASData& blasData);
        void applyResolvedMaterials();
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
    };
}

#endif //COLLECTION_SCENELOADER_HPP