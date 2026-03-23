//
// Created by igor on 10/17/25.
//

#ifndef COLLECTION_PATHTRACER_HPP
#define COLLECTION_PATHTRACER_HPP

#include "Triangle.hpp"
//#include "AbsCamera.hpp"
#include "Camera.hpp"

#include <vector>
#include <memory>

#include "BVH.hpp"

namespace crv::graphics {
    namespace cs = scene;

    template <typename Node, typename Primitive>
    class PathTracer {
    public:
        using BvhType = BVH<Node, Primitive>;
        using Type = BvhType::Node::Type;
        using Vec3 = glm::vec<3, Type, glm::defaultp>;
        using PreTri = PrecomputedTriangle<Type>;

        PathTracer() = default;
        PathTracer(const BvhType& mBvh, const scene::CameraCreateInfo& createInfo);

        scene::AbsCamera* camera() const { return mCamera.get(); }

        std::vector<uint8_t> render() const;
    private:
        Vec3 traceRay(const Ray<Type>& ray) const;

        std::unique_ptr<scene::AbsCamera> mCamera;
        BvhType mBvh;
    };


    static uint8_t floatToByte(const float f) {
        return static_cast<uint8_t>(glm::clamp(f, 0.0f, 1.0f) * 255.0f);
    }

    static void setColor( std::vector<uint8_t>& buffer, const int idx, const auto& color) {
        buffer[idx + 0] = floatToByte(color.r);
        buffer[idx + 1] = floatToByte(color.g);
        buffer[idx + 2] = floatToByte(color.b);
    }

    template <typename Node, typename Primitive>
    PathTracer<Node, Primitive>::PathTracer(const BvhType& bvh, const scene::CameraCreateInfo& createInfo): mBvh(bvh) {
        mCamera = cs::makeCameraUnique(createInfo);
    }

    template <typename Node, typename Primitive>
    std::vector<uint8_t> PathTracer<Node, Primitive>::render() const {
        constexpr int width = 800;
        const int height = static_cast<int>(std::round(width / mCamera->aspectRatio()));
        const float imagePlaneHeight = 2.0f * tan(glm::radians(mCamera->FOV() * 0.5f));
        const float imagePlaneWidth  = imagePlaneHeight * mCamera->aspectRatio();
        std::vector<uint8_t> imageBuffer;
        imageBuffer.resize(width * height * 3);
        for (int i = 0; i < width; ++i) {
            const float u = (static_cast<float>(i) + 0.5f) / width;
            const float px = (2.0f * u - 1.0f) * imagePlaneWidth * 0.5f;
            for (int j = 0; j < height; ++j) {
                const float v = (static_cast<float>(j) + 0.5f) / height;
                const float py = (1.0f - 2.0f * v) * imagePlaneHeight * 0.5f;
                glm::vec3 dir = glm::normalize(px * mCamera->right() + py * mCamera->up() + mCamera->forward());
                Vec3 color = traceRay( {mCamera->position(), dir, mCamera->nearPlane(), mCamera->farPlane()} );
                setColor(imageBuffer, (width * j + i) * 3, color);
            }
        }
        return imageBuffer;
    }

    template<typename Node, typename Primitive>
    PathTracer<Node, Primitive>::Vec3 PathTracer<Node, Primitive>::traceRay(const Ray<Type> &ray) const {
        auto hit = mBvh.intersect(ray, 1e-6);
        if (!hit) return {0, 0, 0};
        auto& [N, t, u, v] = *hit;
        Vec3 P = ray.pos + ray.dir * t;
        constexpr Vec3 L = {1, 2, 3};
        Type I = std::max(static_cast<Type>(0), glm::dot(N, L));
        Vec3 color = {90, 90, 90};
        return 255 * I * color;
    }
}

#endif //COLLECTION_PATHTRACER_HPP
