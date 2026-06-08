//
// Created by igor on 6/8/26.
//

#include "AccelerationStructure.hpp"

namespace crv::graphics::vulkan {
    AccelerationStructure::AccelerationStructure(const BLASCreateInfo& info):
    mDevice(info.device) {
        create(info);
    }

    AccelerationStructure::AccelerationStructure(const TLASCreateInfo& info):
    mDevice(info.device) {
        create(info);
    }

    VkDeviceAddress AccelerationStructure::deviceAddress() const {
        const VkAccelerationStructureDeviceAddressInfoKHR info{
            .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR,
            .accelerationStructure = mHandle,
        };
        LOAD_VK_FN(mDevice, vkGetAccelerationStructureDeviceAddressKHR);
        return vkGetAccelerationStructureDeviceAddressKHR(mDevice, &info);
    }

    void AccelerationStructure::create(const ASCreateInfo& info) {
        VkAccelerationStructureBuildGeometryInfoKHR buildInfo {
            .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
            .type = info.type,
            .flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR,
            .geometryCount = 1,
            .pGeometries = &info.geometry,
        };
        VkAccelerationStructureBuildSizesInfoKHR sizeInfo {
            .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR,
        };

        LOAD_VK_FN(mDevice, vkGetAccelerationStructureBuildSizesKHR);
        vkGetAccelerationStructureBuildSizesKHR(mDevice, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
            &buildInfo, &info.primitiveCount, &sizeInfo);

        const BufferCreateInfo bufferCreateInfo {
            .allocator = info.allocator,
            .size = sizeInfo.accelerationStructureSize,
            .bufferUsage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
                           VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .memoryUsage = VMA_MEMORY_USAGE_AUTO
        };
        mBuffer = Buffer(bufferCreateInfo);

        const VkAccelerationStructureCreateInfoKHR asCreateInfo{
            .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR,
            .buffer = mBuffer.get(),
            .offset = 0,
            .size = sizeInfo.accelerationStructureSize,
            .type = info.type
        };
        LOAD_VK_FN(mDevice, vkCreateAccelerationStructureKHR);
        vkCreateAccelerationStructureKHR(mDevice, &asCreateInfo, nullptr, &mHandle);

        VkPhysicalDeviceAccelerationStructurePropertiesKHR asProps {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR
        };
        VkPhysicalDeviceProperties2 props2 {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
            .pNext = &asProps
        };
        vkGetPhysicalDeviceProperties2(info.physicalDevice, &props2);

        const BufferCreateInfo scratchBufferCreateInfo {
            .allocator = info.allocator,
            .size = sizeInfo.buildScratchSize,
            .bufferUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                           VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .memoryUsage = VMA_MEMORY_USAGE_AUTO,
            .minAlignment = asProps.minAccelerationStructureScratchOffsetAlignment
        };
        mScratchBuffer = Buffer(scratchBufferCreateInfo);
        buildInfo.dstAccelerationStructure = mHandle;
        buildInfo.scratchData.deviceAddress = mScratchBuffer.deviceAddress(mDevice);
        VkAccelerationStructureBuildRangeInfoKHR rangeInfo{.primitiveCount = info.primitiveCount};
        const VkAccelerationStructureBuildRangeInfoKHR* ranges[] = {&rangeInfo};
        LOAD_VK_FN(mDevice, vkCmdBuildAccelerationStructuresKHR);
        vkCmdBuildAccelerationStructuresKHR(info.commandBuffer, 1, &buildInfo, ranges);

        VkMemoryBarrier2 barrier {
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
            .srcStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
            .srcAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
            .dstStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
            .dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR,
        };
        const VkDependencyInfo dependency {
            .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .memoryBarrierCount = 1,
            .pMemoryBarriers = &barrier,
        };
        vkCmdPipelineBarrier2(info.commandBuffer, &dependency);
    }

    void AccelerationStructure::create(const BLASCreateInfo& info) {
        const VkAccelerationStructureGeometryTrianglesDataKHR triangles {
            .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR,
            .vertexFormat = VK_FORMAT_R32G32B32_SFLOAT,
            .vertexData = {.deviceAddress = info.vertexAddress},
            .vertexStride = info.vertexStride,
            .maxVertex = info.vertexCount - 1,
            .indexType = VK_INDEX_TYPE_UINT32,
            .indexData = {.deviceAddress = info.indexAddress},
        };

        VkAccelerationStructureGeometryKHR geometry {
            .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
            .geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR,
            .flags = VK_GEOMETRY_OPAQUE_BIT_KHR,
        };
        geometry.geometry.triangles = triangles;

        const ASCreateInfo asCreateInfo {
            .type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
            .geometry = geometry,
            .commandBuffer = info.commandBuffer,
            .physicalDevice = info.physicalDevice,
            .allocator = info.allocator,
            .primitiveCount = info.indexCount / 3
        };
        create(asCreateInfo);
    }

    void AccelerationStructure::create(const TLASCreateInfo& info) {
        VkAccelerationStructureGeometryInstancesDataKHR instancesData {
            .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR,
            .arrayOfPointers = VK_FALSE,
        };
        instancesData.data.deviceAddress = info.instanceAddress;

        VkAccelerationStructureGeometryKHR geometry {
            .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
            .geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR,
        };
        geometry.geometry.instances = instancesData;

        const ASCreateInfo asCreateInfo {
            .type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,
            .geometry = geometry,
            .commandBuffer = info.commandBuffer,
            .physicalDevice = info.physicalDevice,
            .allocator = info.allocator,
            .primitiveCount = info.instanceCount
        };
        create(asCreateInfo);
    }

    void AccelerationStructure::destroy() {
        if (mDevice == VK_NULL_HANDLE or mHandle == VK_NULL_HANDLE) return;
        LOAD_VK_FN(mDevice, vkDestroyAccelerationStructureKHR);
        vkDestroyAccelerationStructureKHR(mDevice, mHandle, nullptr);
        mHandle = VK_NULL_HANDLE;
        mDevice = VK_NULL_HANDLE;
        mBuffer.destroy();
    }
}