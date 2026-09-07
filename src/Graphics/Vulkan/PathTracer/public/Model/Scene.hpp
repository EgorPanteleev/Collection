//
// Created by igor on 6/10/26.
//

#ifndef COLLECTION_SCENE_HPP
#define COLLECTION_SCENE_HPP

#include <nlohmann/json.hpp>
using json = nlohmann::json;

#include "Loader.hpp"
#include "Message.hpp"
#include "Timer.hpp"
#include "CoreUtils.hpp"
#include "Types.hpp"
#include "Model/MeshData.hpp"

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

    struct SceneCreateInfo {
        Context* context = nullptr;
    };

    class Scene {
    public:
        Scene() = default;
        explicit Scene(const SceneCreateInfo& info);
        void load(const json& scene);
        [[nodiscard]] json save() const;
    private:
        void loadModel(uint32_t modelIndex, const std::string& path);
        void loadMaterials();
        void applyResolvedMaterials();
        void loadResolvedTexture(const json& jm, const char* key, int textureType,
                                 uint32_t& texIndex, std::string& texName, std::string& texPath);
        void loadExplicitInstances();

        json     mJson{};
        bool     mExplicit = false;
    public:
        DirectLight                  mDirectLight{};
        glm::vec3                    mSkyColor{0.1f};
        std::vector<MeshData>        mMeshes{};
        std::vector<InstanceData>    mInstances{};
        std::vector<uint32_t>        mEmissiveIndices{};
        std::vector<Material>        mMaterials{};
        std::vector<cm::Texture>     mTextureSources{};
        uint32_t                     mSkyboxIndex = UINT32_MAX;
        std::string                  mSkyboxName{};
        std::string                  mSkyboxPath{};
    };
}

#endif //COLLECTION_SCENE_HPP