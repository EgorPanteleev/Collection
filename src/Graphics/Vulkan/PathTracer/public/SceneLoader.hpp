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
    private:
        void loadModel(uint32_t modelIndex, const std::string& path);
        void loadMaterials();

        json     mJson{};
        Context* mContext = nullptr;
    public:
        DirectLight                  mDirectLight{};
        std::vector<BLASData>        mBLASDatas{};
        std::vector<InstanceData>    mInstances{};
        std::vector<InstanceData>    mEmissiveInstances{};
        std::vector<Material>        mMaterials{};
        std::vector<Texture>         mTextures{};
    };
}

#endif //COLLECTION_SCENELOADER_HPP