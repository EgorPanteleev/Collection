//
// Created by igor on 4/26/26.
//

#ifndef COLLECTION_GRAPHICSPIPELINES_HPP
#define COLLECTION_GRAPHICSPIPELINES_HPP

#include "DefaultWrapper.hpp"

namespace crv::graphics::vulkan {
    struct GraphicsPipelinesCreateInfo {
        VkDevice device = VK_NULL_HANDLE;
        std::vector<VkPipelineLayout> layouts{};
        VkFormat colorFormat = VK_FORMAT_UNDEFINED;
        VkVertexInputBindingDescription bindingDescription{};
        std::vector<VkVertexInputAttributeDescription> attributeDescriptions{};
        std::vector<VkPipelineShaderStageCreateInfo> stages{};
    };

    class GraphicsPipelines: public VectorWrapper<VkPipeline> {
    public:
        using VectorWrapper::VectorWrapper;
        explicit GraphicsPipelines(const GraphicsPipelinesCreateInfo& info);
        GraphicsPipelines& operator=(GraphicsPipelines&&) = default;
        ~GraphicsPipelines() override { GraphicsPipelines::destroy(); }
        void destroy() override;
    protected:
        VkDevice mDevice = VK_NULL_HANDLE;
    };
}

#endif //COLLECTION_GRAPHICSPIPELINES_HPP