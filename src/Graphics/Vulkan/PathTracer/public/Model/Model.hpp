//
// Created by igor on 6/12/26.
//

#ifndef COLLECTION_MODEL_HPP
#define COLLECTION_MODEL_HPP

#include "Camera.hpp"
#include "Model/Scene.hpp"
#include "Model/RenderSettings.hpp"
#include "Model/Selection.hpp"
#include "Model/UpdateState.hpp"

namespace crv::graphics::vulkan {
    namespace cs = scene;

    class Model {
    public:
        Model() = default;
        void load(const json& scene);
        [[nodiscard]] Scene&          scene()         { return mScene; }
        [[nodiscard]] RenderSettings& settings()      { return mSettings; }
        [[nodiscard]] Selection&      selection()     { return mSelection; }
        [[nodiscard]] UpdateState&    updateState()   { return mUpdateState; }
        [[nodiscard]] cs::FlyCamera&      flyCamera()     { return mFlyCamera; }
        [[nodiscard]] cs::OrbitalCamera&  orbitalCamera() { return mOrbitalCamera; }
        [[nodiscard]] cs::AbsCamera*      camera() const  { return mCamera; }
        void setCamera(cs::AbsCamera* camera) { mCamera = camera; }

        void setActiveCamera(cs::CameraType type);

        uint32_t addMaterial(uint32_t instanceIndex);
        void duplicateInstances(const std::vector<uint32_t>& indices);
        void removeInstances(const std::vector<uint32_t>& indices);
        void addTexture(const std::string& path, uint32_t materialIndex, int textureType);
        void loadSkybox(const std::string& path);
        void removeSkybox();

        void clearSelection();
        void selectInstance(uint32_t index, bool additive) { select(index + 1, additive); }
        void select(uint32_t id, bool additive);
        void regionSelect(int x0, int y0, int x1, int y1, bool additive, uint32_t width, uint32_t height);
    private:
        Scene             mScene{};
        RenderSettings    mSettings{};
        Selection         mSelection{};
        UpdateState       mUpdateState{};
        cs::FlyCamera     mFlyCamera{};
        cs::OrbitalCamera mOrbitalCamera{};
        cs::AbsCamera*    mCamera = nullptr;
    };
}

#endif //COLLECTION_MODEL_HPP
