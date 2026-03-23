//
// Created by igor on 10/17/25.
//

#ifndef COLLECTION_PATHTRACER_HPP
#define COLLECTION_PATHTRACER_HPP

#include "Triangle.hpp"
#include "Camera.hpp"
#include "BVH.hpp"
#include "ThreadPool.hpp"
#include "Loader.hpp"
#include "Message.hpp"

#include <vector>
#include <memory>

namespace crv::graphics {
    namespace cs = scene;
    namespace cm = model;

    template <typename Node, typename Primitive>
    struct PathTracerCreateInfo {
        cs::AbsCamera* camera;
        cm::Loader* loader;
        BVH<Node, Primitive>* bvh;
        std::vector<size_t>* materialIndices;
        int width;
        int height;
    };

    template <typename Node, typename Primitive>
    class PathTracer {
    public:
        using BvhType = BVH<Node, Primitive>;
        using Type = BvhType::Node::Type;
        using Vec3 = glm::vec<3, Type, glm::defaultp>;
        using PreTri = PrecomputedTriangle<Type>;

        PathTracer() = default;
        PathTracer(const PathTracerCreateInfo<Node, Primitive>& createInfo);

        std::vector<uint8_t> render() const;
        std::vector<uint8_t> render_parallel() const;
    private:
        Vec3 traceRay(const Ray<Type>& ray) const;

        cs::AbsCamera* mCamera;
        cm::Loader* mLoader;
        BvhType* mBvh;
        std::vector<size_t>* mMaterialIndices;
        int mWidth;
        int mHeight;
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
    PathTracer<Node, Primitive>::PathTracer(const PathTracerCreateInfo<Node, Primitive>& createInfo):
    mCamera(createInfo.camera), mLoader(createInfo.loader),
    mBvh(createInfo.bvh), mMaterialIndices(createInfo.materialIndices),
    mWidth(createInfo.width), mHeight(createInfo.height) {}

    template <typename Node, typename Primitive>
    std::vector<uint8_t> PathTracer<Node, Primitive>::render() const {
        const float imagePlaneHeight = 2.0f * tan(glm::radians(mCamera->FOV() * 0.5f));
        const float imagePlaneWidth  = imagePlaneHeight * mCamera->aspectRatio();
        std::vector<uint8_t> imageBuffer;
        imageBuffer.resize(mWidth * mHeight * 3);
        for (int i = 0; i < mWidth; ++i) {
            const float u = (static_cast<float>(i) + 0.5f) / mWidth;
            const float px = (2.0f * u - 1.0f) * imagePlaneWidth * 0.5f;
            for (int j = 0; j < mHeight; ++j) {
                const float v = (static_cast<float>(j) + 0.5f) / mHeight;
                const float py = (1.0f - 2.0f * v) * imagePlaneHeight * 0.5f;
                glm::vec3 dir = glm::normalize(px * mCamera->right() + py * mCamera->up() + mCamera->forward());
                Vec3 color = traceRay( {mCamera->position(), dir, mCamera->nearPlane(), mCamera->farPlane()} );
                setColor(imageBuffer, (mWidth * j + i) * 3, color);
            }
        }
        return imageBuffer;
    }

    template <typename Node, typename Primitive>
    std::vector<uint8_t> PathTracer<Node, Primitive>::render_parallel() const {
        static constexpr int TILE_SIZE = 32;
        static constexpr int NUM_THREADS = 16;
        const float imagePlaneHeight = 2.0f * tan(glm::radians(mCamera->FOV() * 0.5f));
        const float imagePlaneWidth  = imagePlaneHeight * mCamera->aspectRatio();
        std::vector<uint8_t> imageBuffer;
        imageBuffer.resize(mWidth * mHeight * 3);
        auto task = [&](int x, int y) {
            const int xMax = std::min(x + TILE_SIZE, mWidth);
            const int yMax = std::min(y + TILE_SIZE, mHeight);
            for (int i = x; i < xMax; ++i) {
                const float u = (static_cast<float>(i) + 0.5f) / mWidth;
                const float px = (2.0f * u - 1.0f) * imagePlaneWidth * 0.5f;
                for (int j = y; j < yMax; ++j) {
                    const float v = (static_cast<float>(j) + 0.5f) / mHeight;
                    const float py = (1.0f - 2.0f * v) * imagePlaneHeight * 0.5f;
                    glm::vec3 dir = glm::normalize(px * mCamera->right() + py * mCamera->up() + mCamera->forward());
                    Vec3 color = traceRay( {mCamera->position(), dir, mCamera->nearPlane(), mCamera->farPlane()} );
                    setColor(imageBuffer, (mWidth * j + i) * 3, color);
                }
            }
        };
        ThreadPool pool(NUM_THREADS);
        for (int i = 0; i < mWidth; i += TILE_SIZE) {
            for (int j = 0; j < mHeight; j += TILE_SIZE) {
                pool.enqueue([&task, i, j](){ task(i, j); });
            }
        }
        pool.wait();
        return imageBuffer;
    }

    template<typename Node, typename Primitive>
    PathTracer<Node, Primitive>::Vec3 PathTracer<Node, Primitive>::traceRay(const Ray<Type> &ray) const {
        auto hit = mBvh->intersect(ray, 1e-6);
        if (!hit) return {0, 0, 0};
        auto& [id, t, u, v] = *hit;
        const Primitive& primitive = mBvh->primitive(id);
        const cm::Material& material = mLoader->materials()[mMaterialIndices->at(id)];
        auto& texData = material.mTextures[cm::Texture::DIFFUSE].mDataByLevel[0];
        auto uv = u * mLoader->vertices()[mLoader->indices()[id * 3 + 1]].texCoord0 +
            v * mLoader->vertices()[mLoader->indices()[id * 3 + 2]].texCoord0 +
            (1.0f - u - v) * mLoader->vertices()[mLoader->indices()[id * 3 + 0]].texCoord0;
        uv.x = std::clamp(uv.x, 0.0f, 1.0f);
        uv.y = std::clamp(uv.y, 0.0f, 1.0f);

        int x = static_cast<int>(uv.x * (texData.width - 1));
        int y = static_cast<int>(uv.y * (texData.height - 1));
        int channels = 4;
        int texIdx = (y * texData.width + x) * channels;
        auto imgData = static_cast<unsigned char*>(texData.data);
        unsigned char r = imgData[texIdx + 0];
        unsigned char g = imgData[texIdx + 1];
        unsigned char b = imgData[texIdx + 2];

        Vec3 N = primitive.normal();
        Vec3 P = ray.pos + ray.dir * t;
        constexpr Vec3 L = {1, 2, 3};
        Type I = std::max(static_cast<Type>(0), glm::dot(N, L));
        Vec3 color = {r / 255.0, g / 255.0, b / 255.0};

        return color;
    }
}

#endif //COLLECTION_PATHTRACER_HPP
