//
// Created by igor on 5/11/26.
//

#ifndef COLLECTION_OUTLINER_HPP
#define COLLECTION_OUTLINER_HPP

#include "Context.hpp"
#include "DescriptorSetLayout.hpp"
#include "DescriptorPool.hpp"
#include "DescriptorSets.hpp"
#include "PipelineLayout.hpp"
#include "ShaderModule.hpp"
#include "GraphicsPipelines.hpp"

namespace crv::graphics::vulkan {
    struct OutlinerCreateInfo {
        Context* context = nullptr;
        uint32_t framesInFlight = 1;
    };

    struct OutlinerUpdateInfo {
        VkImageView tracerImageView     = VK_NULL_HANDLE;
        VkImageView instanceIdImageView = VK_NULL_HANDLE;
        VkSampler   sampler             = VK_NULL_HANDLE;
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
        void createDescriptorSetLayout();
        void createDescriptorPool();
        std::vector<VkDescriptorSetLayout> getDescriptorLayouts();
        void createDescriptorSets();
        void createPipelineLayout();
        void createShaders();
        void createGraphicsPipelines();
        //void createBuffers(const OutlinerCreateInfo& info);

        Context* mContext = nullptr;
        DescriptorSetLayout mDescriptorSetLayout{};
        DescriptorPool mDescriptorPool{};
        DescriptorSets mDescriptorSets{};
        PipelineLayout mPipelineLayout{};
        ShaderModule mVertexShader{};
        ShaderModule mFragmentShader{};
        GraphicsPipelines mGraphicsPipelines{};
        uint32_t mFramesInFlight = 1;
    };
}

#endif //COLLECTION_OUTLINER_HPP