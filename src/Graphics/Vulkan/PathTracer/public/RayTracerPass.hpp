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
#include "TypesGPU.hpp"

namespace crv::graphics::vulkan {
    namespace cs = scene;

    struct RayTracerPassCreateInfo {
        Context*                      context           = nullptr;
        std::vector<BLASInfo>*        blasInfos         = nullptr;
        AccelerationStructure*        tlas              = nullptr;
        std::vector<InstanceInfo>*    instanceInfos     = nullptr;
        std::vector<Texture>*         textures          = nullptr;
        std::vector<Material>*        materials         = nullptr;
        ImageView*                    outView           = nullptr;
        ImageView*                    outInstanceIdView = nullptr;
        uint32_t                      framesInFlight    = 0;
    };

    struct RayTracerPassUpdateInfo {
        cs::AbsCamera* camera       = nullptr;
        DirectLightGPU directLight{};
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
        void updateMaterial(uint32_t index);
        void updateInstance(uint32_t index);
    private:
        void createDescriptorManager();
        void createShaders();
        void createPipelineLayout();
        void createPipelines();
        void createSBT();
        void createBuffers();

        uint32_t                        mFramesInFlight       = 0;

        Context*                        mContext              = nullptr;
        std::vector<BLASInfo>*          mBLASInfos            = nullptr;
        AccelerationStructure*          mTLAS                 = nullptr;
        std::vector<InstanceInfo>*      mInstanceInfos        = nullptr;
        std::vector<Texture>*           mTextures             = nullptr;
        std::vector<Material>*          mMaterials            = nullptr;
        ImageView*                      mOutputView           = nullptr;
        ImageView*                      mOutputInstanceIdView = nullptr;
        DescriptorManager               mDescriptorManager{};
        PipelineLayout                  mPipelineLayout       = CRV_NULL_HANDLE;
        RayTracerPipelines              mPipelines            = CRV_NULL_HANDLE;
        Buffer                          mSBTBuffer            = CRV_NULL_HANDLE;
        VkStridedDeviceAddressRegionKHR mRaygenRegion{};
        VkStridedDeviceAddressRegionKHR mMissRegion{};
        VkStridedDeviceAddressRegionKHR mHitRegion{};
        VkStridedDeviceAddressRegionKHR mCallRegion{};

        ShaderModule                    mRaygenShader         = CRV_NULL_HANDLE;
        ShaderModule                    mMissShader           = CRV_NULL_HANDLE;
        ShaderModule                    mShadowMissShader     = CRV_NULL_HANDLE;
        ShaderModule                    mHitShader            = CRV_NULL_HANDLE;

        Buffer                          mBLASInfoBuffer{};
        Buffer                          mInstanceInfoBuffer{};
        Buffer                          mMaterialBuffer{};
        std::vector<Buffer>             mCameraBuffers{};
        std::vector<Buffer>             mDirectLightBuffers{};
    };
}

#endif //COLLECTION_RAYTRACERPASS_HPP