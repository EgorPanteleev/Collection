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
#include "Model/MeshData.hpp"
#include "Model/InstanceData.hpp"
#include "Model/Material.hpp"
#include "Model/DirectLight.hpp"

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

        [[nodiscard]] const std::vector<MeshData>&     meshes()         const { return mMeshes; }
        [[nodiscard]] const std::vector<InstanceData>& instances()      const { return mInstances; }
        [[nodiscard]] const std::vector<Material>&     materials()      const { return mMaterials; }
        [[nodiscard]] const std::vector<cm::Texture>&  textureSources() const { return mTextureSources; }
        [[nodiscard]] const std::vector<uint32_t>&     emissiveIndices() const { return mEmissiveIndices; }
        [[nodiscard]] uint32_t           skyboxIndex() const { return mSkyboxIndex; }
        [[nodiscard]] const std::string& skyboxName()  const { return mSkyboxName; }
        [[nodiscard]] const std::string& skyboxPath()  const { return mSkyboxPath; }
        [[nodiscard]] const glm::vec3&   skyColor()    const { return mSkyColor; }
        [[nodiscard]] const DirectLight& directLight() const { return mDirectLight; }

        uint32_t addMaterial(const Material& material);
        void     setMaterial(uint32_t index, const Material& material);
        void     setMaterialTexture(uint32_t materialIndex, int textureType,
                                    uint32_t texIndex, const std::string& name, const std::string& path);
        uint32_t addTextureSource(cm::Texture texture);

        void addModel(const std::string& path);
        void addInstance(const InstanceData& instance);
        [[nodiscard]] std::vector<uint32_t> duplicateInstances(const std::vector<uint32_t>& indices);
        void removeInstances(const std::vector<uint32_t>& indices);
        void removeInstance(uint32_t index);
        void setInstanceTransform(uint32_t index, const Transform& transform);
        void setInstanceName(uint32_t index, const std::string& name);
        void setInstanceMaterial(uint32_t instanceIndex, uint32_t materialIndex);

        void setSkybox(uint32_t index, const std::string& name, const std::string& path);
        void clearSkybox();
        void setSkyColor(const glm::vec3& color);
        void setDirectLight(const DirectLight& light);

        void recomputeEmissiveIndices();
        void recomputeWorlds();
    protected:
        void loadModel(uint32_t modelIndex, const std::string& path);
        void buildMeshes(cm::Loader& loader, uint32_t modelIndex);
        void buildInstances(cm::Loader& loader, uint32_t modelIndex, uint32_t meshBase, uint32_t materialBase);
        void addNode(const cm::Node& node, int32_t parentIndex, uint32_t meshBase,
                     uint32_t materialBase, cm::Loader& loader, uint32_t materialOverride,
                     const glm::mat4& accum = glm::mat4(1.0f));
        void loadModelMaterials(cm::Loader& loader);
        void loadJsonMaterials();
        void applyResolvedMaterials();
        void loadResolvedTexture(const json& jm, const char* key, int textureType,
                                 uint32_t& texIndex, std::string& texName, std::string& texPath);
        void loadExplicitInstances();

        json                         mJson{};
        bool                         mExplicit = false;
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