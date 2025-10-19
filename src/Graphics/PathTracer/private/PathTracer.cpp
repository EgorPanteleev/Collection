//
// Created by igor on 10/17/25.
//

#include "PathTracer.hpp"
#include "Camera.hpp"

namespace crv::graphics {

    static uint8_t floatToByte(float f) {
        return static_cast<uint8_t>(glm::clamp(f, 0.0f, 1.0f) * 255.0f);
    }

    static void setColor( std::vector<uint8_t>& buffer, int idx, const PathTracer::Vec3& color) {
        buffer[idx + 0] = floatToByte(color.r);
        buffer[idx + 1] = floatToByte(color.g);
        buffer[idx + 2] = floatToByte(color.b);
    }

    PathTracer::PathTracer(const std::vector<PreTri>& mTriangles, const scene::CameraCreateInfo& createInfo ):
    mTriangles(mTriangles) {
        mCamera = scene::makeCameraUnique(createInfo);
    }

    std::vector<uint8_t> PathTracer::render() const {
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
                setColor(imageBuffer, (width * j + i) * 3, color);
            }
        }
        return imageBuffer;
    }

    PathTracer::Vec3 PathTracer::traceRay(const Ray<Type>& ray) const {
        for (const auto& sphere: mTriangles) {
            if (!sphere.intersect(ray, 0.01)) continue;
            return {1,0,0};
        }
        return {0,0,0};
    }
}