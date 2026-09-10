//
// Created by igor on 6/7/26.
//

#ifndef COLLECTION_PATHTRACERAPP_HPP
#define COLLECTION_PATHTRACERAPP_HPP

#include "View/Renderer.hpp"
#include "Model/Model.hpp"
#include "InputState.hpp"
#include "CommandStream.hpp"
#include "InputHandlers/CameraInputHandler.hpp"
#include "InputHandlers/AppInputHandler.hpp"

#include <memory>
#include <string>

namespace crv::graphics::vulkan {
    struct PathTracerAppCreateInfo {
        std::string scenePath{};
    };

    class PathTracerApp {
        friend class AppInputHandler;
    public:
        PathTracerApp() = delete;
        explicit PathTracerApp(const PathTracerAppCreateInfo& createInfo);
        void run();
        [[nodiscard]] Window& window() { return mRenderer->window(); }
        [[nodiscard]] InputState& input() { return mInput; }
        [[nodiscard]] CommandStream& commands() { return mCommands; }
    private:
        void applyCommands(double deltaTime);

        Model                     mModel{};
        std::unique_ptr<Renderer> mRenderer{};
        InputState                mInput{};
        CommandStream             mCommands{};
        CameraInputHandler        mCameraHandler{};
        AppInputHandler           mAppHandler{};
    };
}

#endif //COLLECTION_PATHTRACERAPP_HPP
