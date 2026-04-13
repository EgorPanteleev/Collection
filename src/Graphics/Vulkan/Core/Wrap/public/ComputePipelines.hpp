//
// Created by igor on 4/13/26.
//

#ifndef COLLECTION_COMPUTEPIPELINES_HPP
#define COLLECTION_COMPUTEPIPELINES_HPP

#include "DefaultWrapper.hpp"

namespace crv::graphics::vulkan {
    struct ComputePipelinesCreateInfo {
        VkDevice device = VK_NULL_HANDLE;
        std::vector<VkPipelineShaderStageCreateInfo> stages;
        std::vector<VkPipelineLayout> layouts;
    };

    class ComputePipelines: public VectorWrapper<VkPipeline> {
        public:
        using VectorWrapper::VectorWrapper;
        explicit ComputePipelines(const ComputePipelinesCreateInfo& info);
        ComputePipelines& operator=(ComputePipelines&&) = default;
        ~ComputePipelines() override { ComputePipelines::destroy(); }
        void destroy() override;
        protected:
        VkDevice mDevice = VK_NULL_HANDLE;
    };
}


#endif //COLLECTION_COMPUTEPIPELINES_HPP