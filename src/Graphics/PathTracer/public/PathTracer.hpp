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

        static uint8_t floatToByte(float f) {
        return static_cast<uint8_t>(glm::clamp(f, 0.0f, 1.0f) * 255.0f);
    }

    template <typename Node, typename Primitive>
    static void setColor( std::vector<uint8_t>& buffer, int idx, const typename PathTracer<Node, Primitive>::Vec3& color) {
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
        int width = 800;
        int height = static_cast<int>(std::round(width / mCamera->aspectRatio()));
        float imagePlaneHeight = 2.0f * tan(glm::radians(mCamera->FOV() * 0.5f));
        float imagePlaneWidth  = imagePlaneHeight * mCamera->aspectRatio();
        std::vector<uint8_t> imageBuffer;
        imageBuffer.resize( width * height * 3 );
        for (int i = 0; i < width; ++i) {
            float u = (i + 0.5f) / width;
            float px = (2.0f * u - 1.0f) * imagePlaneWidth * 0.5f;
            for (int j = 0; j < height; ++j) {
                float v = (j + 0.5f) / height;
                float py = (1.0f - 2.0f * v) * imagePlaneHeight * 0.5f;
                glm::vec3 dir = glm::normalize(px * mCamera->right() + py * mCamera->up() + mCamera->forward());
                Vec3 color = traceRay( {mCamera->position(), dir, mCamera->nearPlane(), mCamera->farPlane()} );
                setColor<Node, Primitive>(imageBuffer, (width * j + i) * 3, color);
            }
        }
        return imageBuffer;
    }

    template<typename Node, typename Primitive>
    PathTracer<Node, Primitive>::Vec3 PathTracer<Node, Primitive>::traceRay(const Ray<Type> &ray) const {
        auto hit = mBvh.intersect(ray, 1e-6);
        if (!hit) return {0, 0, 0};
        return {std::get<0>(*hit), std::get<1>(*hit), std::get<2>(*hit)};
    }
}

#endif //COLLECTION_PATHTRACER_HPP
