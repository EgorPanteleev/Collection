//
// Created by igor on 4/26/26.
//

#ifndef COLLECTION_GRAPHICSPIPELINES_HPP
#define COLLECTION_GRAPHICSPIPELINES_HPP

#include "DefaultWrapper.hpp"

namespace crv::graphics::vulkan {
    struct GraphicsPipelineCreateInfo {
        VkDevice device = VK_NULL_HANDLE;
        VkPipelineLayout layout{};
        std::vector<VkFormat> colorFormats{};
        VkVertexInputBindingDescription bindingDescription{};
        std::vector<VkVertexInputAttributeDescription> attributeDescriptions{};
        std::vector<VkPipelineShaderStageCreateInfo> stages{};
    };

    class GraphicsPipelines: public VectorWrapper<VkPipeline> {
    public:
        using VectorWrapper::VectorWrapper;
        explicit GraphicsPipelines(const GraphicsPipelineCreateInfo& info);
        explicit GraphicsPipelines(const std::vector<GraphicsPipelineCreateInfo>& infos);
        GraphicsPipelines& operator=(GraphicsPipelines&&) = default;
        ~GraphicsPipelines() override { GraphicsPipelines::destroy(); }
        void destroy() override;
    protected:
        VkDevice mDevice = VK_NULL_HANDLE;
    };
}

#endif //COLLECTION_GRAPHICSPIPELINES_HPP