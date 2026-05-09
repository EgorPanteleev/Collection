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
#include "BinnedSAHBuilder.hpp"
#include "Light.hpp"

namespace crv::app {
    namespace cg = graphics;
    namespace cs = scene;
    namespace cm = model;
    namespace cu = utils;

    template <typename T>
    struct PathTracerAppCreateInfo {
        cs::CameraCreateInfo cameraCreateInfo;
        glm::mat4 model = glm::mat4(1.);
        std::vector<cg::Light<T>*> lights;
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
        PathTracerApp(const PathTracerAppCreateInfo<Type>& createInfo);
        void run();
        void quit() const { mWindow.close(); }
        cs::AbsCamera* camera() { return mCamera.get(); }
        static const char* title() { return "Path Tracer"; }
        friend void mouseMoveCallback<Type, primBits>(GLFWwindow*, double, double);
    private:
        void loadModel(const glm::mat4& modelMatrix, const std::string& modelPath);
        void buildBVH(std::span<Primitive> primitives);

        cm::Loader mLoader;
        std::vector<size_t> mMaterialIndices;
        std::vector<Primitive> mPrimitives;
        cg::BVH16<Node, Primitive> mBvh;
        std::unique_ptr<scene::AbsCamera> mCamera;
        cg::PathTracer<Node, Primitive> mPathTracer;
        cg::Window mWindow;
    };

    static bool rightMouseButtonPressed = false;
    static double lastX = 0.0f, lastY = 0.0f;

    static void processKeyboard(GLFWwindow* window, cs::AbsCamera* camera, double deltaTime) {
        auto speed = 0.03 * deltaTime;
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
    static void scrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
        auto app = static_cast<PathTracerApp<Type, primBits>*>(glfwGetWindowUserPointer(window));
        cs::AbsCamera* camera = app->camera();
        float speed = 10.0f;
        camera->zoom(yoffset * speed);
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
    PathTracerApp<Type, primBits>::PathTracerApp(const PathTracerAppCreateInfo<Type>& createInfo):
    mCamera(cs::makeCameraUnique(createInfo.cameraCreateInfo)), mWindow(title(), createInfo.width, createInfo.height) {
        mWindow.init();
        loadModel(createInfo.model, createInfo.modelPath);
        cg::PathTracerCreateInfo<Node, Primitive> pathTracerCreateInfo = {
            .camera = mCamera.get(),
            .loader = &mLoader,
            .bvh = &mBvh,
            .lights = createInfo.lights,
            .materialIndices = &mMaterialIndices,
            .width = createInfo.width,
            .height = createInfo.height
        };
        mPathTracer = {pathTracerCreateInfo};
    }

    inline void add(std::vector<uint16_t>& buf1, const std::vector<uint8_t>& buf2) {
        for (int i = 0; i < buf1.size(); ++i) buf1[i] += buf2[i];
    }

    inline void divide(std::vector<uint8_t>& buf1, const int cnt) {
        for (int i = 0; i < buf1.size(); ++i) buf1[i] /= cnt;
    }

    template <typename Type, size_t primBits>
    void PathTracerApp<Type, primBits>::run() {
        std::vector<uint8_t> imageBuffer;
        imageBuffer.resize(mWindow.width() * mWindow.height() * 3);
        std::vector<uint8_t> currBuffer;
        currBuffer.resize(mWindow.width() * mWindow.height() * 3);
        mWindow.makeContextCurrent();
        mWindow.setUserPoint(this);
        mWindow.setKeyCallBack(keyCallBack<Type, primBits>);
        mWindow.setMouseButtonCallBack(mouseButtonCallback);
        mWindow.setMouseMoveCallBack(mouseMoveCallback<Type, primBits>);
        mWindow.setScrollCallBack(scrollCallback<Type, primBits>);

        if ( !initGLEW() ) return;
        const GLuint tex = createTexture(mWindow.width(), mWindow.height(), imageBuffer.data());
        GLuint VAO, VBO, EBO;
        createBuffers(VAO, VBO, EBO);
        const GLuint shader = createShaderProgram();
        utils::FpsCounter fpsCounter;
        double deltaTime = 0;
        int cnt = 0;
        auto oldState = mCamera->state();
        auto currState = oldState;
        while(!mWindow.shouldClose()) {
            fpsCounter.update();
            std::string newTitle(title());
            mWindow.setTitle( (newTitle + "(" + fpsCounter.fpsAsString() + " fps)").c_str() );
            deltaTime = 1e3 / fpsCounter.fps();
            currState = mCamera->state();

            if (oldState != currState) {
                oldState = currState;
                cnt = 0;
            }
            currBuffer = mPathTracer.render_parallel();

            for (int i = 0; i < imageBuffer.size(); ++i) {
                imageBuffer[i] = (imageBuffer[i] * cnt + currBuffer[i]) / (cnt + 1);
            }
            ++cnt;
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
    void PathTracerApp<Type, primBits>::loadModel(const glm::mat4& modelMatrix, const std::string& modelPath) {
        cu::Timer timer;
        timer.start();
        mLoader.setModel(modelPath);
        mLoader.load(modelMatrix);
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
        INFO << "Total number of primitives: " << mPrimitives.size();
        buildBVH(std::span(mPrimitives));
    }

    template <typename Type, size_t primBits>
    void PathTracerApp<Type, primBits>::buildBVH(std::span<Primitive> primitives) {
        cu::Timer timer;
        timer.start();
        cg::BinnedSAHBuilder<Node, Primitive> builder{ primitives };
        mBvh = builder.build();
        INFO << "BVH build time: " << timer.duration() / 1000 << " sec";
    }
}

#endif //CPUPATHTRACER_HPP
