//
// Created by igor on 5/11/26.
//

#ifndef COLLECTION_OUTLINER_HPP
#define COLLECTION_OUTLINER_HPP

#include "Context.hpp"
#include "DescriptorManager.hpp"
#include "PipelineLayout.hpp"
#include "ShaderModule.hpp"
#include "GraphicsPipelines.hpp"

namespace crv::graphics::vulkan {
    struct OutlinerCreateInfo {
        Context*    context             = nullptr;
        VkImageView tracerImageView     = VK_NULL_HANDLE;
        std::vector<VkImageView> instanceIdImageViews{};
        VkSampler   tracerSampler       = VK_NULL_HANDLE;
        std::vector<VkSampler> instanceIdSamplers{};
        uint32_t    framesInFlight      = 1;
    };

    struct OutlinerUpdateInfo {
    };

    struct OutlinerRecordInfo {
        VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
        VkImageView     outImageView  = VK_NULL_HANDLE;
        VkExtent2D      extent{};
        uint32_t        currentFrame  = 0;
    };

    class Outliner {
    public:
        Outliner() = default;
        explicit Outliner(const OutlinerCreateInfo& info);
        void update(const OutlinerUpdateInfo& info);
        void record(const OutlinerRecordInfo& info);
    protected:
        void createDescriptorManager(const OutlinerCreateInfo& info);
        void createPipelineLayout();
        void createShaders();
        void createGraphicsPipelines();
        //void createBuffers(const OutlinerCreateInfo& info);

        Context* mContext = nullptr;
        DescriptorManager mDescriptorManager{};
        PipelineLayout mPipelineLayout{};
        ShaderModule mVertexShader{};
        ShaderModule mFragmentShader{};
        GraphicsPipelines mGraphicsPipelines{};
        uint32_t mFramesInFlight = 1;
    };
}

#endif //COLLECTION_OUTLINER_HPP