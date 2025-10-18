//
// Created by igor on 10/18/25.
//

#ifndef CPUPATHTRACER_HPP
#define CPUPATHTRACER_HPP

#include "PathTracer.hpp"
#include "Sphere.hpp"
#include "Window.hpp"

namespace crv::app {
    struct PathTracerAppCreateInfo {
        using Type = float;
        int width;
        int height;
        std::vector<graphics::Sphere<Type>> spheres;
        scene::CameraCreateInfo cameraCreateInfo;
    };

    class PathTracerApp {
    public:
        PathTracerApp(const PathTracerAppCreateInfo& createInfo);
        void run();
    private:
        graphics::PathTracer mPathTracer;
        graphics::Window mWindow;
    };
}

#endif //CPUPATHTRACER_HPP
