//
// Created by igor on 6/7/26.
//

#ifndef COLLECTION_RAYTRACERPASS_HPP
#define COLLECTION_RAYTRACERPASS_HPP

#include "Context.hpp"
#include "DescriptorManager.hpp"
#include "PipelineLayout.hpp"
#include "ShaderModule.hpp"
#include "RayTracerPipelines.hpp"

namespace crv::graphics::vulkan {
    struct RayTracerPassCreateInfo {
        Context*   context        = nullptr;
        ImageView* outView        = nullptr;
        uint32_t   framesInFlight = 0;
    };

    class RayTracerPass {
    public:
        RayTracerPass() = default;
        explicit RayTracerPass(const RayTracerPassCreateInfo& createInfo);
    private:
        void createDescriptorManager();
        void createShaders();
        void createPipelineLayout();
        void createPipelines();
        void createSBT();

        Context*                        mContext           = nullptr;
        ImageView*                      mOutView           = nullptr;
        DescriptorManager               mDescriptorManager{};
        PipelineLayout                  mPipelineLayout    = CRV_NULL_HANDLE;
        RayTracerPipelines              mPipelines         = CRV_NULL_HANDLE;
        Buffer                          mSBTBuffer         = CRV_NULL_HANDLE;
        VkStridedDeviceAddressRegionKHR mRaygenRegion{};
        VkStridedDeviceAddressRegionKHR mMissRegion{};
        VkStridedDeviceAddressRegionKHR mHitRegion{};
        VkStridedDeviceAddressRegionKHR mCallRegion{};

        ShaderModule                    mRaygenShader       = CRV_NULL_HANDLE;
        ShaderModule                    mMissShader         = CRV_NULL_HANDLE;
        ShaderModule                    mHitShader          = CRV_NULL_HANDLE;
        uint32_t                        mFramesInFlight     = 0;
    };
}

#endif //COLLECTION_RAYTRACERPASS_HPP