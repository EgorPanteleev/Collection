//
// Created by igor on 6/7/26.
//

#ifndef COLLECTION_PATHTRACERAPP_HPP
#define COLLECTION_PATHTRACERAPP_HPP

#include <nlohmann/json.hpp>
using json = nlohmann::json;

#include "Context.hpp"
#include "Camera.hpp"
#include "Swapchain.hpp"
#include "Image.hpp"
#include "ImageView.hpp"
#include "RayTracerPass.hpp"

namespace crv::graphics::vulkan {
    namespace cs = scene;

    struct PathTracerAppCreateInfo {
        std::string scenePath{};
    };

    class PathTracerApp {
    public:
        PathTracerApp() = delete;
        explicit PathTracerApp(const PathTracerAppCreateInfo& createInfo);
    private:
        void readScene(const std::string& scenePath);
        void createContext();
        void createSwapChain();
        void createImages();
        void createRayTracerPass();
        void createCamera();
        void createSwapChainImages();

#ifdef NDEBUG
        bool mDebug = false;
#else
        bool mDebug = true;
#endif
        uint32_t               mFramesInFlight = 3;

        json                   mScene{};
        Context                mContext        = CRV_NULL_HANDLE;
        Swapchain              mSwapchain      = CRV_NULL_HANDLE;
        RayTracerPass          mRayTracerPass  = CRV_NULL_HANDLE;

        cs::FlyCamera          mFlyCamera{};
        cs::OrbitalCamera      mOrbitalCamera{};
        cs::AbsCamera*         mCamera         = nullptr;

        std::vector<VkImage>   mSwapchainImages{};
        std::vector<ImageView> mSwapchainImageViews{};
        Image                  mTracerImage    = CRV_NULL_HANDLE;
        ImageView              mTracerView     = CRV_NULL_HANDLE;
    };
}

#endif //COLLECTION_PATHTRACERAPP_HPP