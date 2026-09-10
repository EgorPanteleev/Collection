//
// Created by igor on 6/7/26.
//

#include "PathTracerApp.hpp"
#include "Timer.hpp"
#include "CallBacks.hpp"

#include <fstream>

namespace crv::graphics::vulkan {
    namespace cu = utils;

    namespace {
        json readScene(const std::string& scenePath) {
            std::ifstream file(scenePath);
            json scene;
            file >> scene;
            return scene;
        }

        WindowCreateInfo windowConfig(const json& scene) {
            const auto& window = scene["window"];
            return {
                .width  = window["width"],
                .height = window["height"],
                .name   = window["name"]
            };
        }
    }

    PathTracerApp::PathTracerApp(const PathTracerAppCreateInfo& createInfo) {
        const json scene = readScene(createInfo.scenePath);
        mModel.load(scene);
        RendererCreateInfo rendererCreateInfo {
            .windowCreateInfo = windowConfig(scene),
            .model            = &mModel,
            .commands         = &mCommands
        };
        mRenderer = std::make_unique<Renderer>(rendererCreateInfo);
        setCallBacks(this);
        mRenderer->initUI();
    }

    void PathTracerApp::run() {
        cu::FpsCounter fpsCounter;
        double deltaTime = 0;
        Window& window = mRenderer->window();
        while (!window.shouldClose()) {
            mInput.beginFrame();
            glfwPollEvents(); //todo remove
            window.keyboardCallBack(deltaTime); //todo remove
            applyCommands(deltaTime); //apply commands two times??
            mRenderer->beginFrame();
            applyCommands(deltaTime);
            mRenderer->endFrame();
            fpsCounter.update();
            deltaTime = 1e3 / fpsCounter.fps();
            window.setTitle(std::to_string(fpsCounter.fps()).c_str());
        }
        mRenderer->waitIdle();
    }

    void PathTracerApp::applyCommands(const double deltaTime) {
        bool cameraMoved = false;
        const CameraInput cameraInput {
            .camera    = mModel.camera(),
            .input     = &mInput,
            .deltaTime = static_cast<float>(deltaTime),
        };
        for (const Command& command : mCommands.get()) {
            switch (commandTarget(command.type)) {
                case CommandTarget::CAMERA:
                    if (mCameraHandler.apply(command, cameraInput)) cameraMoved = true;
                    else mAppHandler.apply(command, this);
                    break;
                case CommandTarget::SCENE:
                case CommandTarget::VIEW:
                case CommandTarget::APP:
                    mAppHandler.apply(command, this);
                    break;
                default:
                    break;
            }
        }
        mCommands.clear();
        if (cameraMoved) mRenderer->onCameraMoved();
    }
}
