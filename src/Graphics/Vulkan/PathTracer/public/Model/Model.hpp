//
// Created by igor on 6/12/26.
//

#ifndef COLLECTION_MODEL_HPP
#define COLLECTION_MODEL_HPP

#include "Camera.hpp"
#include "Model/Scene.hpp"
#include "Model/RenderSettings.hpp"
#include "Model/Selection.hpp"

namespace crv::graphics::vulkan {
    namespace cs = scene;

    class Model {
    public:
        [[nodiscard]] Scene&          scene()         { return mScene; }
        [[nodiscard]] RenderSettings& settings()      { return mSettings; }
        [[nodiscard]] Selection&      selection()     { return mSelection; }
        [[nodiscard]] cs::FlyCamera&      flyCamera()     { return mFlyCamera; }
        [[nodiscard]] cs::OrbitalCamera&  orbitalCamera() { return mOrbitalCamera; }
        [[nodiscard]] cs::AbsCamera*      camera() const  { return mCamera; }
        void setCamera(cs::AbsCamera* camera) { mCamera = camera; }
    private:
        Scene             mScene{};
        RenderSettings    mSettings{};
        Selection         mSelection{};
        cs::FlyCamera     mFlyCamera{};
        cs::OrbitalCamera mOrbitalCamera{};
        cs::AbsCamera*    mCamera = nullptr;
    };
}

#endif //COLLECTION_MODEL_HPP
