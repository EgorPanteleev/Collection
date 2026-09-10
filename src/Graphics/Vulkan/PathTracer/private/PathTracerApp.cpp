//
// Created by igor on 6/7/26.
//

#include "PathTracerApp.hpp"
#include "Timer.hpp"
#include "InputCallbacks.hpp"
#include "Keybindings.hpp"

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
        mCommandHandler = CommandHandler(CommandHandlerCreateInfo{
            .model    = &mModel,
            .renderer = mRenderer.get(),
            .input    = &mInput
        });
        setCallBacks(mRenderer->window(), mInput);
        mRenderer->initUI();
    }

    void PathTracerApp::run() {
        cu::FpsCounter fpsCounter;
        double deltaTime = 0;
        Window& window = mRenderer->window();
        while (!window.shouldClose()) {
            mInput.beginFrame();
            glfwPollEvents();
            processInput(mInput, mCommands);
            applyCommands(deltaTime);
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
        mCommandHandler.apply(mCommands, static_cast<float>(deltaTime));
        mCommands.clear();
    }
}
