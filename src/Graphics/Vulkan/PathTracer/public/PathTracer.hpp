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

namespace crv::graphics::vulkan {
    class PathTracer {
    public:
        explicit PathTracer(const WindowCreateInfo& windowCreateInfo);
    protected:
        void createContext(const WindowCreateInfo& windowCreateInfo);
        void createDescriptorSetLayout();
        void createDescriptorPool();
        void createDescriptorSets();
        void createPipelineLayout();
        void createShaders();
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
    };
}

#endif //COLLECTION_PATHTRACER_HPP