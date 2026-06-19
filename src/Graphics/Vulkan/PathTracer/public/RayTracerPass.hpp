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
#include "Types.hpp"
#include "Types.hpp"

namespace crv::graphics::vulkan {
    namespace cs = scene;

    struct RayTracerPassCreateInfo {
        Context*                      context                = nullptr;
        AccelerationStructure*        tlas                   = nullptr;
        Buffer*                       BLASBuffer             = nullptr;
        Buffer*                       instanceBuffer         = nullptr;
        Buffer*                       emissiveInstanceBuffer = nullptr;
        Buffer*                       materialBuffer         = nullptr;
        std::vector<Texture>*         textures               = nullptr;
        ImageView*                    outView                = nullptr;
        ImageView*                    outInstanceIdView      = nullptr;
        uint32_t                      framesInFlight    = 0;
    };

    struct RayTracerPassUpdateInfo {
        cs::AbsCamera* camera       = nullptr;
        DirectLight    directLight{};
        uint32_t       currentFrame = 0;
    };

    struct RayTracerPassRecordInfo {
        VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
        PushConstants   constants{};
        uint32_t        width         = 0;
        uint32_t        height        = 0;
    };

    class RayTracerPass {
    public:
        RayTracerPass() = default;
        explicit RayTracerPass(const RayTracerPassCreateInfo& info);
        void update(const RayTracerPassUpdateInfo& info);
        void record(const RayTracerPassRecordInfo& info);
        void bindTexture(uint32_t index);
    private:
        void createDescriptorManager();
        void createShaders();
        void createPipelineLayout();
        void createPipelines();
        void createSBT();
        void createBuffers();

        uint32_t                        mFramesInFlight         = 0;

        Context*                        mContext                = nullptr;
        AccelerationStructure*          mTLAS                   = nullptr;
        std::vector<Texture>*           mTextures               = nullptr;
        ImageView*                      mOutputView             = nullptr;
        ImageView*                      mOutputInstanceIdView   = nullptr;
        DescriptorManager               mDescriptorManager{};
        PipelineLayout                  mPipelineLayout         = CRV_NULL_HANDLE;
        RayTracerPipelines              mPipelines              = CRV_NULL_HANDLE;
        Buffer                          mSBTBuffer              = CRV_NULL_HANDLE;
        VkStridedDeviceAddressRegionKHR mRaygenRegion{};
        VkStridedDeviceAddressRegionKHR mMissRegion{};
        VkStridedDeviceAddressRegionKHR mHitRegion{};
        VkStridedDeviceAddressRegionKHR mCallRegion{};

        ShaderModule                    mRaygenShader           = CRV_NULL_HANDLE;
        ShaderModule                    mMissShader             = CRV_NULL_HANDLE;
        ShaderModule                    mShadowMissShader       = CRV_NULL_HANDLE;
        ShaderModule                    mHitShader              = CRV_NULL_HANDLE;

        Buffer*                         mBLASBuffer             = nullptr;
        Buffer*                         mInstanceBuffer         = nullptr;
        Buffer*                         mEmissiveInstanceBuffer = nullptr;
        Buffer*                         mMaterialBuffer         = nullptr;
        std::vector<Buffer>             mCameraBuffers{};
        std::vector<Buffer>             mDirectLightBuffers{};
    };
}

#endif //COLLECTION_RAYTRACERPASS_HPP