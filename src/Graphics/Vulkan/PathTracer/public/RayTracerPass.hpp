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
#include "AccelerationStructure.hpp"
#include "Camera.hpp"

namespace crv::graphics::vulkan {
    namespace cs = scene;

    struct RayTracerPassCreateInfo {
        Context*               context        = nullptr;
        AccelerationStructure* tlas           = nullptr;
        ImageView*             outView        = nullptr;
        uint32_t               framesInFlight = 0;
    };

    struct RayTracerPassUpdateInfo {
        cs::AbsCamera* camera       = nullptr;
        uint32_t       currentFrame = 0;
    };

    struct RayTracerPassRecordInfo {
        VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
        uint32_t        width         = 0;
        uint32_t        height        = 0;
    };

    class RayTracerPass {
    public:
        RayTracerPass() = default;
        explicit RayTracerPass(const RayTracerPassCreateInfo& info);
        void update(const RayTracerPassUpdateInfo& info);
        void record(const RayTracerPassRecordInfo& info);
    private:
        void createDescriptorManager();
        void createShaders();
        void createPipelineLayout();
        void createPipelines();
        void createSBT();
        void createBuffers();
        uint32_t                        mFramesInFlight     = 0;

        Context*                        mContext           = nullptr;
        AccelerationStructure*          mTLAS              = nullptr;
        ImageView*                      mOutputView        = nullptr;
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

        std::vector<Buffer>             mCameraBuffers{};
    };
}

#endif //COLLECTION_RAYTRACERPASS_HPP