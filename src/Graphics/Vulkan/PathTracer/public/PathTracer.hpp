//
// Created by igor on 4/8/26.
//

#ifndef COLLECTION_PATHTRACER_HPP
#define COLLECTION_PATHTRACER_HPP

#include "Context.hpp"
#include "DescriptorSetLayout.hpp"
#include "DescriptorPool.hpp"
#include "DescriptorSets.hpp"
#include "PipelineLayout.hpp"
#include "ShaderModule.hpp"
#include "ComputePipelines.hpp"
#include "CommandPool.hpp"
#include "CommandBuffers.hpp"
#include "Buffer.hpp"

namespace crv::graphics::vulkan {
    class PathTracer {
    public:
        explicit PathTracer(const WindowCreateInfo& windowCreateInfo);
        void run();
    protected:
        void createContext(const WindowCreateInfo& windowCreateInfo);
        void createDescriptorSetLayout();
        void createDescriptorPool();
        void createDescriptorSets();
        void createPipelineLayout();
        void createShaders();
        void createComputePipelines();
        void createCommandPool();
        void createCommandBuffers();
        void update();
        void record();
        void submit();

        [[nodiscard]] std::vector<VkDescriptorSetLayout> getDescriptorLayouts() const;
        [[nodiscard]] std::vector<VkPipelineLayout> getPipelineLayouts() const;
#ifdef NDEBUG
        bool mDebug = false;
#else
        bool mDebug = true;
#endif
        Context mContext{};
        DescriptorSetLayout mDescriptorSetLayout{};
        DescriptorPool mDescriptorPool{};
        DescriptorSets mDescriptorSets{};
        PipelineLayout mPipelineLayout{};
        ShaderModule mShader{};
        ComputePipelines mComputePipelines{};
        CommandPool mCommandPool{};
        CommandBuffers mCommandBuffers{};
        Buffer mBuffer1{};
        Buffer mBuffer2{};
    };
}

#endif //COLLECTION_PATHTRACER_HPP