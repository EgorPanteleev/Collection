//
// Created by igor on 10/17/25.
//

#ifndef COLLECTION_PATHTRACER_HPP
#define COLLECTION_PATHTRACER_HPP

#include "Triangle.hpp"
#include "AbsCamera.hpp"

#include <vector>
#include <memory>

namespace crv::graphics {
    class PathTracer {
    public:
        using Type = float;
        using Vec3 = glm::vec<3, Type, glm::defaultp>;
        using PreTri = PrecomputedTriangle<Type>;

        PathTracer() = default;
        PathTracer(const std::vector<PreTri>& mTriangles, const scene::CameraCreateInfo& createInfo );

        scene::AbsCamera* camera() const { return mCamera.get(); }

        std::vector<uint8_t> render() const;
    private:
        Vec3 traceRay(const Ray<Type>& ray) const;

        std::unique_ptr<scene::AbsCamera> mCamera;
        std::vector<PreTri> mTriangles;
    };
}

#endif //COLLECTION_PATHTRACER_HPP
