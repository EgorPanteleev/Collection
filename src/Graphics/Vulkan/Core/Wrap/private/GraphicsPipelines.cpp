//
// Created by igor on 4/26/26.
//

#include "GraphicsPipelines.hpp"
#include "Message.hpp"

namespace crv::graphics::vulkan {
    GraphicsPipelines::GraphicsPipelines(const GraphicsPipelinesCreateInfo &info): mDevice(info.device) {
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

        VkFormat colorFormats[] = { info.colorFormat };

        VkPipelineRenderingCreateInfo pipelineRenderingInfo {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
                .colorAttachmentCount = 1,
                .pColorAttachmentFormats = colorFormats,
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

        VkPipelineInputAssemblyStateCreateInfo inputAssemblyInfo{
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

        VkPipelineRasterizationStateCreateInfo rasterizationInfo {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                .depthClampEnable = VK_FALSE,
                .rasterizerDiscardEnable = VK_FALSE,
                .polygonMode = VK_POLYGON_MODE_FILL,
                .cullMode = VK_CULL_MODE_BACK_BIT,
                .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
                .depthBiasEnable = VK_FALSE,
                .depthBiasConstantFactor = 0.0f,
                .depthBiasClamp = 0.0f,
                .depthBiasSlopeFactor = 0.0f,
                .lineWidth = 1.0f
        };

        VkPipelineMultisampleStateCreateInfo multisampleInfo{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
                .sampleShadingEnable = VK_TRUE,
                .minSampleShading = .2f,
                .pSampleMask = nullptr,
                .alphaToCoverageEnable = VK_FALSE,
                .alphaToOneEnable = VK_FALSE
        };

        // VkPipelineColorBlendAttachmentState colorBlendAttachment{
        //         .blendEnable = VK_FALSE,
        // };

        VkPipelineColorBlendAttachmentState colorBlendAttachment{
            .blendEnable = VK_TRUE,
            .srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA,
            .dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
            .colorBlendOp = VK_BLEND_OP_ADD,
            .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
            .dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
            .alphaBlendOp = VK_BLEND_OP_ADD,
            .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                              VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT
        };

        VkPipelineColorBlendStateCreateInfo colorBlendInfo{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
                .logicOpEnable = VK_FALSE,
                .logicOp = VK_LOGIC_OP_COPY,
                .attachmentCount = 1,
                .pAttachments = &colorBlendAttachment,
                .blendConstants = { 0.0f, 0.0f, 0.0f, 0.0f }
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

        std::vector<VkGraphicsPipelineCreateInfo> pipelineInfos;
        for (size_t i = 0; i < info.layouts.size(); ++i) {
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
                .layout = info.layouts[i],
                .renderPass = VK_NULL_HANDLE,
                .subpass = 0,
                .basePipelineHandle = VK_NULL_HANDLE,
                .basePipelineIndex = -1
            };
            pipelineInfos.push_back(pipelineInfo);
        }

    // VkPipelineCacheCreateInfo cacheCreateInfo = {
    //         .sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO,
    //         .initialDataSize = 0,
    //         .pInitialData = nullptr,
    // };
    // vkCreatePipelineCache(mContext->device(), &cacheCreateInfo, nullptr, &mPipelineCache);

        mVec.resize(info.layouts.size());
        if (vkCreateGraphicsPipelines(info.device, nullptr, pipelineInfos.size(), pipelineInfos.data(), nullptr, mVec.data()) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create graphics pipeline!");
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
