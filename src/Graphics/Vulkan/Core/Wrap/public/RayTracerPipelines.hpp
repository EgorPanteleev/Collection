//
// Created by igor on 6/7/26.
//

#ifndef COLLECTION_RAYTRACERPIPELINES_HPP
#define COLLECTION_RAYTRACERPIPELINES_HPP

#include "DefaultWrapper.hpp"

namespace crv::graphics::vulkan {
    struct RayTracerPipelinesCreateInfo {
        VkDevice                                          device = VK_NULL_HANDLE;
        std::vector<VkPipelineShaderStageCreateInfo>      stages{};
        std::vector<VkRayTracingShaderGroupCreateInfoKHR> groups{};
        VkPipelineLayout                                  layout = VK_NULL_HANDLE;
    };

    class RayTracerPipelines: public VectorWrapper<VkPipeline> {
    public:
        using VectorWrapper::VectorWrapper;
        explicit RayTracerPipelines(const RayTracerPipelinesCreateInfo& info);
        RayTracerPipelines& operator=(RayTracerPipelines&&) = default;
        ~RayTracerPipelines() override { RayTracerPipelines::destroy(); }
        void destroy() override;
    protected:
        VkDevice mDevice = VK_NULL_HANDLE;
    };
}

#endif //COLLECTION_RAYTRACERPIPELINES_HPP