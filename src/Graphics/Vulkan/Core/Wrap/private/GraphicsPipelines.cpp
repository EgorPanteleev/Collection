//
// Created by igor on 4/26/26.
//

#include "GraphicsPipelines.hpp"
#include "Message.hpp"

namespace crv::graphics::vulkan {
    GraphicsPipelines::GraphicsPipelines(const GraphicsPipelineCreateInfo& info): GraphicsPipelines(std::vector{info}) {}
    GraphicsPipelines::GraphicsPipelines(const std::vector<GraphicsPipelineCreateInfo>& infos): mDevice(infos[0].device) {
        VkPipelineDepthStencilStateCreateInfo depthStencil {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
                .depthTestEnable = VK_TRUE,
                .depthWriteEnable = VK_TRUE,
                .depthCompareOp = VK_COMPARE_OP_LESS,
                .depthBoundsTestEnable = VK_FALSE,
                .stencilTestEnable = VK_FALSE,
                .front = {},
                .back = {},
                .minDepthBounds = 0.0f,
                .maxDepthBounds = 1.0f
        };

        VkPipelineInputAssemblyStateCreateInfo inputAssemblyInfo {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
            .primitiveRestartEnable = VK_FALSE
        };

        VkPipelineViewportStateCreateInfo viewportInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            .viewportCount = 1,
            .pViewports = nullptr,
            .scissorCount = 1,
            .pScissors = nullptr
        };

        VkPipelineRasterizationStateCreateInfo rasterizationInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            .depthClampEnable = VK_FALSE,
            .rasterizerDiscardEnable = VK_FALSE,
            .polygonMode = VK_POLYGON_MODE_FILL,
            .cullMode = VK_CULL_MODE_NONE,
            .depthBiasEnable = VK_FALSE,
            .depthBiasConstantFactor = 0.0f,
            .depthBiasClamp = 0.0f,
            .depthBiasSlopeFactor = 0.0f,
            .lineWidth = 1.0f
        };

        VkPipelineMultisampleStateCreateInfo multisampleInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
            .sampleShadingEnable = VK_FALSE,
        };

        std::vector dynamicStateEnables = {
            VK_DYNAMIC_STATE_VIEWPORT,
            VK_DYNAMIC_STATE_SCISSOR
        };

        VkPipelineDynamicStateCreateInfo dynamicStateInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
            .dynamicStateCount = static_cast<uint32_t>(dynamicStateEnables.size()),
            .pDynamicStates = dynamicStateEnables.data()
        };

        mVec.resize(infos.size());
        for (size_t i = 0; i < infos.size(); ++i) {
            const auto& info = infos[i];
            VkPipelineRenderingCreateInfo pipelineRenderingInfo {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
                .colorAttachmentCount = static_cast<uint32_t>(info.colorFormats.size()),
                .pColorAttachmentFormats = info.colorFormats.data(),
                .depthAttachmentFormat = VK_FORMAT_D32_SFLOAT,
                .stencilAttachmentFormat = VK_FORMAT_UNDEFINED,
            };

            VkPipelineVertexInputStateCreateInfo vertexInputInfo{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
                .vertexBindingDescriptionCount = 1,
                .pVertexBindingDescriptions = &info.bindingDescription,
                .vertexAttributeDescriptionCount = static_cast<uint32_t>(info.attributeDescriptions.size()),
                .pVertexAttributeDescriptions = info.attributeDescriptions.data()
            };

            VkPipelineColorBlendAttachmentState colorBlendAttachment{
                .blendEnable = VK_FALSE,
                .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                  VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT
            };
            std::vector colorBlendAttachments(info.colorFormats.size(), colorBlendAttachment);

            VkPipelineColorBlendStateCreateInfo colorBlendInfo{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
                .logicOpEnable = VK_FALSE,
                .logicOp = VK_LOGIC_OP_COPY,
                .attachmentCount = static_cast<uint32_t>(colorBlendAttachments.size()),
                .pAttachments = colorBlendAttachments.data(),
                .blendConstants = { 0.0f, 0.0f, 0.0f, 0.0f }
            };
            VkGraphicsPipelineCreateInfo pipelineInfo{
                .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
                .pNext = &pipelineRenderingInfo,
                .stageCount = 2,
                .pStages = info.stages.data(),
                .pVertexInputState = &vertexInputInfo,
                .pInputAssemblyState = &inputAssemblyInfo,
                .pViewportState = &viewportInfo,
                .pRasterizationState = &rasterizationInfo,
                .pMultisampleState = &multisampleInfo,
                .pDepthStencilState = &depthStencil,
                .pColorBlendState = &colorBlendInfo,
                .pDynamicState = &dynamicStateInfo,
                .layout = info.layout,
                .renderPass = VK_NULL_HANDLE,
                .subpass = 0,
                .basePipelineHandle = VK_NULL_HANDLE,
                .basePipelineIndex = -1
            };
            if (vkCreateGraphicsPipelines(mDevice, nullptr, 1, &pipelineInfo, nullptr, &mVec[i]) != VK_SUCCESS) {
                throw std::runtime_error("Failed to create graphics pipeline!");
            }
        }

        INFO << "Created graphics pipeline!";
    }

    void GraphicsPipelines::destroy() {
        if (mDevice == VK_NULL_HANDLE or mVec.empty()) return;
        for (auto& pipeline: mVec) {
            vkDestroyPipeline(mDevice, pipeline, nullptr);
            pipeline = VK_NULL_HANDLE;
        }
        mVec.clear();
        mDevice = VK_NULL_HANDLE;
    }
}
