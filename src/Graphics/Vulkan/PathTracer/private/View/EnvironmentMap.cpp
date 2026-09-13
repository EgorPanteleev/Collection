//
// Created by igor on 6/12/26.
//

#include "View/EnvironmentMap.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace crv::graphics::vulkan {
    namespace {
        constexpr double ENV_PI = 3.14159265358979323846;

        float halfToFloat(uint16_t h) {
            const uint32_t sign = (h >> 15) & 0x1u;
            uint32_t       exp  = (h >> 10) & 0x1Fu;
            uint32_t       mant = h & 0x3FFu;
            uint32_t       bits;
            if (exp == 0) {
                if (mant == 0) {
                    bits = sign << 31;
                } else {
                    exp = 127 - 15 + 1;
                    while ((mant & 0x400u) == 0) { mant <<= 1; --exp; }
                    mant &= 0x3FFu;
                    bits = (sign << 31) | (exp << 23) | (mant << 13);
                }
            } else if (exp == 0x1Fu) {
                bits = (sign << 31) | (0xFFu << 23) | (mant << 13);
            } else {
                bits = (sign << 31) | ((exp - 15 + 127) << 23) | (mant << 13);
            }
            float out;
            std::memcpy(&out, &bits, sizeof(out));
            return out;
        }

        struct EnvDistribution {
            uint32_t           width  = 0;
            uint32_t           height = 0;
            float              integral = 0.0f;
            std::vector<float> condFunc;
            std::vector<float> condCdf;
            std::vector<float> marginalCdf;
        };

        EnvDistribution buildEnvDistribution2D(const uint16_t* pixels, uint32_t width, uint32_t height) {
            EnvDistribution d;
            d.width  = width;
            d.height = height;
            const uint32_t nu = width;
            const uint32_t nv = height;
            d.condFunc.resize(static_cast<size_t>(nu) * nv);
            d.condCdf.resize(static_cast<size_t>(nu + 1) * nv);
            std::vector<float> marginalFunc(nv);

            auto sanitize = [](float v) {
                if (std::isnan(v)) return 0.0f;
                return std::clamp(v, 0.0f, 65504.0f);
            };
            double sum = 0.0;
            for (uint32_t y = 0; y < nv; ++y) {
                const float sinTheta = static_cast<float>(std::sin(ENV_PI * (y + 0.5) / nv));
                float* func = &d.condFunc[static_cast<size_t>(y) * nu];
                for (uint32_t x = 0; x < nu; ++x) {
                    const uint16_t* p = pixels + (static_cast<size_t>(y) * nu + x) * 4;
                    const float lum = 0.2126f * sanitize(halfToFloat(p[0]))
                                    + 0.7152f * sanitize(halfToFloat(p[1]))
                                    + 0.0722f * sanitize(halfToFloat(p[2]));
                    func[x] = lum * sinTheta;
                    sum += func[x];
                }
            }

            const float average = static_cast<float>(sum / (static_cast<double>(nu) * nv));
            for (float& f : d.condFunc) f = std::max(f - average, 0.0f);

            for (uint32_t y = 0; y < nv; ++y) {
                const float* func = &d.condFunc[static_cast<size_t>(y) * nu];
                float* cdf = &d.condCdf[static_cast<size_t>(y) * (nu + 1)];
                cdf[0] = 0.0f;
                for (uint32_t x = 1; x <= nu; ++x) cdf[x] = cdf[x - 1] + func[x - 1] / nu;
                const float funcInt = cdf[nu];
                if (funcInt == 0.0f) {
                    for (uint32_t x = 1; x <= nu; ++x) cdf[x] = static_cast<float>(x) / nu;
                } else {
                    for (uint32_t x = 1; x <= nu; ++x) cdf[x] /= funcInt;
                }
                marginalFunc[y] = funcInt;
            }

            d.marginalCdf.resize(nv + 1);
            d.marginalCdf[0] = 0.0f;
            for (uint32_t y = 1; y <= nv; ++y)
                d.marginalCdf[y] = d.marginalCdf[y - 1] + marginalFunc[y - 1] / nv;
            const float marginalInt = d.marginalCdf[nv];
            if (marginalInt == 0.0f) {
                for (uint32_t y = 1; y <= nv; ++y) d.marginalCdf[y] = static_cast<float>(y) / nv;
            } else {
                for (uint32_t y = 1; y <= nv; ++y) d.marginalCdf[y] /= marginalInt;
            }
            d.integral = marginalInt;
            return d;
        }
    }

    void EnvironmentMap::disable() {
        mMarginalCdfAddr = 0;
        mCondCdfAddr     = 0;
        mCondFuncAddr    = 0;
        mIntegral        = 0.0f;
    }

    void EnvironmentMap::build(const cm::Texture& skybox) {
        disable();
        if (skybox.mDataByLevel.empty() || skybox.mFormat != cm::Texture::R16G16B16A16_SFLOAT) return;
        const auto& level  = skybox.mDataByLevel[0];
        const auto* pixels = static_cast<const uint16_t*>(level.data);
        if (pixels == nullptr || level.width == 0 || level.height == 0) return;

        const EnvDistribution dist = buildEnvDistribution2D(pixels, level.width, level.height);
        if (dist.integral <= 0.0f) return;

        auto fillStorageBuffer = [this](Buffer& dst, const std::vector<float>& data) {
            const size_t size = sizeof(float) * data.size();
            const BufferCreateInfo createInfo{
                .allocator = mContext->allocator(),
                .size = size,
                .bufferUsage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                .memoryUsage = VMA_MEMORY_USAGE_GPU_ONLY
            };
            dst = Buffer(createInfo);
            const CopyDataToGPUBufferInfo copyInfo{
                .data = const_cast<float*>(data.data()),
                .size = size,
                .allocator = mContext->allocator(),
                .buffer = dst.get(),
                .device = mContext->device(),
                .queueFamilyIndex = mContext->familyIndex(QueueFamilyType::GRAPHICS).value(),
                .queue = mContext->queue(QueueFamilyType::GRAPHICS)
            };
            Buffer::copy(copyInfo);
        };

        fillStorageBuffer(mMarginalCdfBuffer, dist.marginalCdf);
        fillStorageBuffer(mCondCdfBuffer,     dist.condCdf);
        fillStorageBuffer(mCondFuncBuffer,    dist.condFunc);
        mMarginalCdfAddr = mMarginalCdfBuffer.deviceAddress(mContext->device());
        mCondCdfAddr     = mCondCdfBuffer.deviceAddress(mContext->device());
        mCondFuncAddr    = mCondFuncBuffer.deviceAddress(mContext->device());
        mIntegral        = dist.integral;
    }
}
