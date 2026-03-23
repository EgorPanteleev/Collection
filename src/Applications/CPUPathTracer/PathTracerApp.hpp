//
// Created by igor on 10/18/25.
//

#ifndef CPUPATHTRACER_HPP
#define CPUPATHTRACER_HPP

#include "PathTracer.hpp"
#include "Window.hpp"
#include "Node.hpp"
#include "Timer.hpp"
#include "GLUtils.hpp"
#include "Triangle.hpp"
#include "Loader.hpp"
#include "SweepSAHBuilder.hpp"

namespace crv::app {
    namespace cg = graphics;
    namespace cs = scene;
    namespace cm = model;
    namespace cu = utils;

    struct PathTracerAppCreateInfo {
        cs::CameraCreateInfo cameraCreateInfo;
        std::string modelPath;
        int width;
        int height;
    };

    template <typename T, size_t primBits>
    class PathTracerApp {
    public:
        using Type = T;
        using Node = cg::Node<Type, 32, primBits>;
        using Primitive = cg::PrecomputedTriangle<Type>;
        PathTracerApp(const PathTracerAppCreateInfo& createInfo);
        void run();
        void quit() const { mWindow.close(); }
        cs::AbsCamera* camera() { return mCamera.get(); }
        static const char* title() { return "Path Tracer"; }
        friend void mouseMoveCallback<Type, primBits>(GLFWwindow*, double, double);
    private:
        void loadModel(const std::string& modelPath);
        void buildBVH(std::span<Primitive> primitives);

        cm::Loader mLoader;
        std::vector<size_t> mMaterialIndices;
        std::vector<Primitive> mPrimitives;
        cg::BVH<Node, Primitive> mBvh;
        std::unique_ptr<scene::AbsCamera> mCamera;
        cg::PathTracer<Node, Primitive> mPathTracer;
        cg::Window mWindow;
    };

    static bool rightMouseButtonPressed = false;
    static double lastX = 0.0f, lastY = 0.0f;

    static void processKeyboard(GLFWwindow* window, cs::AbsCamera* camera, double deltaTime) {
        auto speed = 1;
        //if (speed < 0) return;
        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
            camera->move(speed, 0, 0);
        }
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
            camera->move(-speed, 0, 0);
        }
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
            camera->move(0, -speed, 0);
        }
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
            camera->move(0, speed, 0);
        }
        if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS) {
            camera->move(0, 0, -speed);
        }
        if (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS) {
            camera->move(0, 0, speed);
        }
        float rotateSpeed = speed * 0.3f;

        if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS) {
            camera->rotate(0, rotateSpeed, 0);
        }
        if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS) {
            camera->rotate(0, -rotateSpeed, 0);
        }
        if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS) {
            camera->rotate(rotateSpeed, 0, 0);
        }
        if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS) {
            camera->rotate(-rotateSpeed, 0, 0);
        }
    }

    template <typename Type, size_t primBits>
    static void keyCallBack(GLFWwindow* window, int key, int scancode, int action, int mods) {
        auto app = static_cast<PathTracerApp<Type, primBits>*>(glfwGetWindowUserPointer(window));

        if (action == GLFW_PRESS && key == GLFW_KEY_Q) {
            app->quit();
        }

    }

    static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
        if (button == GLFW_MOUSE_BUTTON_RIGHT) {
            if (action == GLFW_PRESS) {
                rightMouseButtonPressed = true;
                glfwGetCursorPos(window, &lastX, &lastY);
            } else if (action == GLFW_RELEASE) {
                rightMouseButtonPressed = false;
            }
        }
    }

    template <typename Type, size_t primBits>
    void mouseMoveCallback(GLFWwindow* window, double xpos, double ypos) {
        auto app = static_cast<PathTracerApp<Type, primBits>*>(glfwGetWindowUserPointer(window));
        cs::AbsCamera* camera = app->camera();

        if (!rightMouseButtonPressed || !camera) return;

        double sensitivity = 0.1f;
        double offsetX = xpos - lastX;
        double offsetY = lastY - ypos;

        lastX = xpos;
        lastY = ypos;
        camera->rotate(static_cast<float>(-offsetY * sensitivity),
                       static_cast<float>(-offsetX * sensitivity), 0.f);
    }

    template <typename Type, size_t primBits>
    PathTracerApp<Type, primBits>::PathTracerApp(const PathTracerAppCreateInfo& createInfo):
    mCamera(cs::makeCameraUnique(createInfo.cameraCreateInfo)), mWindow(title(), createInfo.width, createInfo.height) {
        loadModel(createInfo.modelPath);
        cg::PathTracerCreateInfo<Node, Primitive> pathTracerCreateInfo = {
            .camera = mCamera.get(),
            .loader = &mLoader,
            .bvh = &mBvh,
            .materialIndices = &mMaterialIndices,
            .width = createInfo.width,
            .height = createInfo.height
        };
        mPathTracer = {pathTracerCreateInfo};
    }

    template <typename Type, size_t primBits>
    void PathTracerApp<Type, primBits>::run() {
        std::vector<uint8_t> imageBuffer;
        mWindow.makeContextCurrent();
        mWindow.setUserPoint(this);
        mWindow.setKeyCallBack(keyCallBack<Type, primBits>);
        mWindow.setMouseButtonCallBack(mouseButtonCallback);
        mWindow.setMouseMoveCallBack(mouseMoveCallback<Type, primBits>);

        if ( !initGLEW() ) return;
        const GLuint tex = createTexture(mWindow.width(), mWindow.height(), imageBuffer.data());
        GLuint VAO, VBO, EBO;
        createBuffers(VAO, VBO, EBO);
        const GLuint shader = createShaderProgram();
        utils::FpsCounter fpsCounter;
        double deltaTime = 0;
        while(!mWindow.shouldClose()) {
            fpsCounter.update();
            std::string newTitle(title());
            mWindow.setTitle( (newTitle + "(" + fpsCounter.fpsAsString() + " fps)").c_str() );
            deltaTime = 1e3 / fpsCounter.fps();
            imageBuffer = mPathTracer.render_parallel();
            updateTexture(tex, mWindow.width(), mWindow.height(), imageBuffer.data());

            int winWidth, winHeight;
            mWindow.getFrameBufferSize(winWidth, winHeight);
            glViewport(0, 0, winWidth, winHeight);

            drawTexture(shader, VAO, tex);

            mWindow.swapBuffers();
            graphics::Window::pollEvents();
            processKeyboard(mWindow.glfwWindow(), mCamera.get(), deltaTime);
        }
        cleanData(tex, shader, VBO, EBO, VAO);
    }

    template <typename Type, size_t primBits>
    void PathTracerApp<Type, primBits>::loadModel(const std::string& modelPath) {
        cu::Timer timer;
        timer.start();
        mLoader.setModel(modelPath);
        mLoader.load();
        INFO << "Model load time: " << timer.duration() / 1000 << " sec";

        timer.start();
        const auto& indices = mLoader.indices();
        const auto& vertices = mLoader.vertices();
        const auto& meshes = mLoader.meshes();
        for (size_t i = 0; i < meshes.size(); ++i) {
            const auto& mesh = meshes[i];
            for (size_t j = 0; j < mesh.numIndices; j += 3) {
                const size_t idx = mesh.baseIndex + j;
                mPrimitives.emplace_back(vertices[indices[idx + 0]].pos,
                                        vertices[indices[idx + 1]].pos,
                                        vertices[indices[idx + 2]].pos);
                mMaterialIndices.emplace_back(mesh.materialIndex);
            }
        }
        INFO << "Primitive creation time: " << timer.duration() / 1000 << " sec";
        buildBVH(std::span(mPrimitives));
    }

    template <typename Type, size_t primBits>
    void PathTracerApp<Type, primBits>::buildBVH(std::span<Primitive> primitives) {
        cu::Timer timer;
        timer.start();
        cg::SweepSAHBuilder<Node, Primitive> builder{ primitives };
        mBvh = builder.build();
        INFO << "BVH build time: " << timer.duration() / 1000 << " sec";
    }
}

#endif //CPUPATHTRACER_HPP
