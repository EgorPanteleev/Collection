//
// Created by igor on 4/13/26.
//

#ifndef COLLECTION_PIPELINELAYOUT_HPP
#define COLLECTION_PIPELINELAYOUT_HPP

#include "DefaultWrapper.hpp"

namespace crv::graphics::vulkan {
    struct PipelineLayoutCreateInfo {
        VkDevice device = VK_NULL_HANDLE;
        std::vector<VkDescriptorSetLayout> layouts{};
        std::vector<VkPushConstantRange> ranges{};
    };

    class PipelineLayout: public DefaultWrapper<VkPipelineLayout> {
    public:
        using DefaultWrapper::DefaultWrapper;
        explicit PipelineLayout(const PipelineLayoutCreateInfo& info);
        PipelineLayout& operator=(PipelineLayout&&) = default;
        ~PipelineLayout() override { PipelineLayout::destroy(); }
        void destroy() override;
    protected:
        VkDevice mDevice = VK_NULL_HANDLE;
    };
}

#endif //COLLECTION_PIPELINELAYOUT_HPP