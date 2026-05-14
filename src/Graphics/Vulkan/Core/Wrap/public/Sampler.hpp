//
// Created by igor on 4/14/26.
//

#ifndef COLLECTION_SAMPLER_HPP
#define COLLECTION_SAMPLER_HPP

#include "DefaultWrapper.hpp"

namespace crv::graphics::vulkan {
    struct SamplerCreateInfo {
        VkDevice             device         = VK_NULL_HANDLE;
        VkPhysicalDevice     physicalDevice = VK_NULL_HANDLE;
        VkSamplerAddressMode addressMode    = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        VkBool32             compareEnable  = VK_FALSE;
        VkCompareOp          compareOp      = VK_COMPARE_OP_LESS_OR_EQUAL;
        uint32_t             mipLevels      = 1;
        VkBorderColor        borderColor    = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
        VkFilter             magFilter      = VK_FILTER_LINEAR;
        VkFilter             minFilter      = VK_FILTER_LINEAR;
        VkSamplerMipmapMode  mipmapMode     = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    };

    class Sampler: public DefaultWrapper<VkSampler> {
    public:
        using DefaultWrapper::DefaultWrapper;
        explicit Sampler(const SamplerCreateInfo& info);
        explicit Sampler(Sampler&&) = default;
        Sampler& operator=(Sampler&&) = default;
        ~Sampler() override { Sampler::destroy(); }
        void destroy() override;
    protected:
        VkDevice mDevice = VK_NULL_HANDLE;
    };
}

#endif //COLLECTION_SAMPLER_HPP