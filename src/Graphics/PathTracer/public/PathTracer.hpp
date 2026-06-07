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
#include "Light.hpp"
#include "Material.hpp"

#include <vector>
#include <memory>

#define MAX_DEPTH 2

namespace crv::graphics {
    namespace cs = scene;
    namespace cm = model;

    template <typename Node, typename Primitive>
    struct PathTracerCreateInfo {
        cs::AbsCamera* camera;
        cm::Loader* loader;
        BVH16<Node, Primitive>* bvh;
        std::vector<Light<typename Node::Type>*> lights;
        std::vector<size_t>* materialIndices;
        int width;
        int height;
    };

    template <typename Node, typename Primitive>
    class PathTracer {
    public:
        using BvhType = BVH16<Node, Primitive>;
        using Type = BvhType::Node::Type;
        using Vec3 = glm::vec<3, Type, glm::defaultp>;
        using Vec2 = glm::vec<2, Type, glm::defaultp>;
        using PreTri = PrecomputedTriangle<Type>;
        using Sample = Light<Type>::Sample;
        using Ray = Ray<Type>;

        PathTracer() = default;
        PathTracer(const PathTracerCreateInfo<Node, Primitive>& createInfo);

        std::vector<uint8_t> render() const;
        std::vector<uint8_t> render_parallel() const;
    private:
        Vec3 traceRay(const Ray& ray, uint8_t depth) const;
        Vec2 getUV(size_t id, Type u, Type v) const;
        Vec3 getColor(size_t id, const Vec2& uv, cm::Texture::Type type) const;
        auto intersect(const Ray& ray, Type eps) const;

        cs::AbsCamera* mCamera;
        cm::Loader* mLoader;
        BvhType* mBvh;
        std::vector<Light<Type>*> mLights;
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
    mBvh(createInfo.bvh), mLights(createInfo.lights), mMaterialIndices(createInfo.materialIndices),
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
                Vec3 color = traceRay( {mCamera->position(), dir, mCamera->nearPlane(), mCamera->farPlane()}, 0 );
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
        auto task = [&](const int x, const int y) {
            const int xMax = std::min(x + TILE_SIZE, mWidth);
            const int yMax = std::min(y + TILE_SIZE, mHeight);
            for (int i = x; i < xMax; ++i) {
                const float u = (static_cast<float>(i) + 0.5f) / mWidth;
                const float px = (2.0f * u - 1.0f) * imagePlaneWidth * 0.5f;
                for (int j = y; j < yMax; ++j) {
                    const float v = (static_cast<float>(j) + 0.5f) / mHeight;
                    const float py = (1.0f - 2.0f * v) * imagePlaneHeight * 0.5f;
                    glm::vec3 dir = glm::normalize(px * mCamera->right() + py * mCamera->up() + mCamera->forward());
                    Vec3 color = traceRay( {mCamera->position(), dir, mCamera->nearPlane(), mCamera->farPlane()}, 0 );
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
        pool.stop();
        return imageBuffer;
    }

    // auto hit = intersect(ray, 1e-6);
    // if (!hit) return {0., 0., 0.};
    // auto& [id, t, u, v] = *hit;
    // Vec2 uv = getUV(id, u, v);
    // Vec3 diffuseColor = getColor(id, uv, cm::Texture::DIFFUSE);
    // const Primitive& primitive = mBvh->primitive(id);
    // Vec3 N = primitive.normal();
    // Vec3 P = ray.pos + ray.dir * t;
    // Vec3 directColor{0};
    // for (auto light: mLights) {
    //     Sample sample = light->sample(P);
    //     Ray shadowRay(P, sample.direction, 1e-3, sample.distance);
    //     auto shadowHit = intersect(shadowRay, 1e-6);
    //     if (shadowHit) continue;
    //     directColor += diffuseColor * sample.radiance;
    // }
    // if (depth == MAX_DEPTH) return directColor;
    // Vec3 wo = mCamera->position() - P;
    // Lambertian<Type> mat;
    // Vec3 wi, brdf;
    // Type pdf;
    // mat.scatter(N, wo, wi, brdf, pdf);
    // Ray nextRay(P, wi, 1e-3, mCamera->farPlane());
    // Vec3 indirectColor = diffuseColor * brdf * traceRay(nextRay, depth + 1) * glm::dot(N, wi) / pdf;
    //
    // return directColor + indirectColor;


    template<typename Node, typename Primitive>
    PathTracer<Node, Primitive>::Vec3 PathTracer<Node, Primitive>::traceRay(const Ray &ray, uint8_t depth) const {
        Vec3 resultColor(0.0f);
        Vec3 throughput(1.0f);
        Ray currentRay = ray;
        for (uint8_t d = depth; d <= MAX_DEPTH; d++) {
            auto hit = intersect(currentRay, 1e-6);
            if (!hit) break;
            auto& [id, t, u, v] = *hit;
            Vec2 uv = getUV(id, u, v);
            Vec3 diffuseColor = getColor(id, uv, cm::Texture::DIFFUSE);
            const Primitive& primitive = mBvh->primitive(id);
            Vec3 N = primitive.normal();
            Vec3 P = currentRay.pos + currentRay.dir * t;

            Vec3 directColor(0.0f);
            for (auto light : mLights) {
                Sample sample = light->sample(P);
                Ray shadowRay(P, sample.direction, 1e-3, sample.distance);
                if (intersect(shadowRay, 1e-6)) continue;
                directColor += diffuseColor * sample.radiance;
            }

            resultColor += throughput * directColor;
            if (d == MAX_DEPTH) break;
            Vec3 wo = -currentRay.dir;
            Lambertian<Type> material;
            Vec3 wi, brdf;
            Type pdf;
            material.scatter(N, wo, wi, brdf, pdf);
            Type cosTheta = glm::dot(N, wi);
            if (pdf <= 0 || cosTheta <= 0) break;
            throughput *= diffuseColor * brdf * cosTheta / pdf;
            currentRay = Ray(P, wi, 1e-3, mCamera->farPlane());
        }
        return resultColor;
    }

    template<typename Node, typename Primitive>
    auto PathTracer<Node, Primitive>::intersect(const Ray& ray, Type eps) const {
        return mBvh->intersect16(ray, eps);
    }

    template<typename Node, typename Primitive>
    PathTracer<Node, Primitive>::Vec2 PathTracer<Node, Primitive>::getUV(const size_t id, Type u, Type v) const {
        const auto& vertices = mLoader->vertices();
        const auto& indices = mLoader->indices();
        const size_t idx = id * 3;
        const Type w = static_cast<Type>(1.) - u - v;
        return u * vertices[indices[idx + 1]].texCoord0 +
               v * vertices[indices[idx + 2]].texCoord0 +
               w * vertices[indices[idx + 0]].texCoord0;
    }

    template<typename Node, typename Primitive>
    PathTracer<Node, Primitive>::Vec3 PathTracer<Node, Primitive>::getColor(const size_t id, const Vec2& uv, cm::Texture::Type type) const {
        const cm::Material& material = mLoader->materials()[(*mMaterialIndices)[id]];
        const auto&[texData, width, height] = material.mTextures[type].mDataByLevel[0];
        const int x = static_cast<int>(uv.x * (width - 1));
        const int y = static_cast<int>(uv.y * (height - 1));
        static constexpr int channels = 4;
        const int texIdx = (y * width + x) * channels;
        const unsigned char* imgData = static_cast<unsigned char*>(texData);
        Type coeff = static_cast<Type>(1.) / 255.;
        return {imgData[texIdx + 0] * coeff, imgData[texIdx + 1] * coeff, imgData[texIdx + 2] * coeff};
    }
}

#endif //COLLECTION_PATHTRACER_HPP
