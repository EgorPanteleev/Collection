//
// Created by igor on 4/14/26.
//

#include "Sampler.hpp"

#include <stdexcept>

namespace crv::graphics::vulkan {
    Sampler::Sampler(const SamplerCreateInfo &info): mDevice(info.device) {
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(info.physicalDevice, &properties);

        const VkSamplerCreateInfo samplerInfo {
            .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
            .magFilter = VK_FILTER_LINEAR,
            .minFilter = VK_FILTER_LINEAR,
            .mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
            .addressModeU = info.addressMode,
            .addressModeV = info.addressMode,
            .addressModeW = info.addressMode,
            .mipLodBias = 0.0f,
            .anisotropyEnable = VK_TRUE,
            .maxAnisotropy = properties.limits.maxSamplerAnisotropy,
            .compareEnable = info.compareEnable,
            .compareOp = info.compareOp,
            .minLod = 0,
            .maxLod = static_cast<float>(info.mipLevels),
            .borderColor = info.borderColor,
            .unnormalizedCoordinates = VK_FALSE,
        };
        if (vkCreateSampler(mDevice, &samplerInfo, nullptr, &mHandle) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create texture sampler!");
        }
    }

    void Sampler::destroy() {
        vkDestroySampler(mDevice, mHandle, nullptr);
    }
}
