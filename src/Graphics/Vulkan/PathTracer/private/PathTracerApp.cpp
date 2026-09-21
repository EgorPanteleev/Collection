//
// Created by igor on 6/7/26.
//

#include "PathTracerApp.hpp"
#include "Timer.hpp"
#include "InputCallbacks.hpp"
#include "Keybindings.hpp"

#include <filesystem>
#include <fstream>

namespace crv::graphics::vulkan {
    namespace cu = utils;

    namespace {
        json emptyScene() {
            return {
                {"modelImports", json::array()},
                {"instances", json::array()},
                {"materials", json::array()},
                {"directLight", {{"direction", {-0.468, 0.318, -0.824}}, {"intensity", 0.0}}},
                {"window", {{"name", "GPU Path Tracer"}, {"width", 1280}, {"height", 720}}},
                {"camera", {
                    {"type", "Fly"},
                    {"position", {0.0, 0.0, 5.0}},
                    {"target", {0.0, 0.0, 0.0}},
                    {"up", {0.0, 1.0, 0.0}},
                    {"zoom", 1.0},
                    {"fov", 60.0},
                    {"nearPlane", 0.1},
                    {"farPlane", 5000.0}
                }}
            };
        }

        json readScene(const std::string& scenePath) {
            if (scenePath.empty()) return emptyScene();
            std::ifstream file(scenePath);
            if (!file) {
                ERROR << "Scene not found: " << scenePath << "; starting from an empty scene";
                return emptyScene();
            }
            json scene = json::parse(file, nullptr, false);
            if (scene.is_discarded()) {
                ERROR << "Scene is not valid JSON: " << scenePath << "; starting from an empty scene";
                return emptyScene();
            }
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
        mModel.setName(std::filesystem::path(createInfo.scenePath).stem().string());
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
            readInput();
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

    void PathTracerApp::readInput() {
        mInput.beginFrame();
        glfwPollEvents();
        processInput(mInput, mCommands);
    }

    void PathTracerApp::applyCommands(const double deltaTime) {
        mCommandHandler.apply(mCommands, static_cast<float>(deltaTime));
        mCommands.clear();
    }
}
