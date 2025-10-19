//
// Created by igor on 10/18/25.
//

#ifndef CPUPATHTRACER_HPP
#define CPUPATHTRACER_HPP

#include "PathTracer.hpp"
#include "Window.hpp"

namespace crv::app {
    struct PathTracerAppCreateInfo {
        using Type = float;
        using PreTri = graphics::PrecomputedTriangle<Type>;
        int width;
        int height;
        std::vector<PreTri> triangles;
        scene::CameraCreateInfo cameraCreateInfo;
    };

    class PathTracerApp {
    public:
        PathTracerApp(const PathTracerAppCreateInfo& createInfo);
        void run();
        void quit() const { mWindow.close(); }
        static const char* title() { return "Path Tracer"; }
        friend void mouseMoveCallback(GLFWwindow*, double, double);
    private:
        graphics::PathTracer mPathTracer;
        graphics::Window mWindow;
    };
}

#endif //CPUPATHTRACER_HPP
