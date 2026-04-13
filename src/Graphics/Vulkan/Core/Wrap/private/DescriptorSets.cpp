//
// Created by igor on 4/13/26.
//

#include "DescriptorSets.hpp"
#include "Message.hpp"

namespace crv::graphics::vulkan {
    DescriptorSets::DescriptorSets(const DescriptorSetsCreateInfo &info) : mDevice(info.device) {
        const uint32_t descriptorSetCount = info.layouts.size();
        VkDescriptorSetVariableDescriptorCountAllocateInfo countAllocInfo{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_ALLOCATE_INFO,
            .descriptorSetCount = descriptorSetCount,
            .pDescriptorCounts = info.variableCounts.data()
        };
        const VkDescriptorSetAllocateInfo allocInfo{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .pNext = &countAllocInfo,
            .descriptorPool = info.pool,
            .descriptorSetCount = descriptorSetCount,
            .pSetLayouts = info.layouts.data()
        };
        mVec.resize(descriptorSetCount);
        if (vkAllocateDescriptorSets(mDevice, &allocInfo, mVec.data()) != VK_SUCCESS) {
            throw std::runtime_error("Failed to allocate descriptor sets!");
        }
        INFO << "Created descriptor sets!";
    }

    void DescriptorSets::update(const DescriptorSetsUpdateInfo &info) const {
        for (size_t i = 0; i < mVec.size(); ++i) {
            auto descriptorWrites = info.descriptorsWrites[i];
            for (auto &descriptorWrite: descriptorWrites) {
                descriptorWrite.dstSet = mVec[i];
            }
            vkUpdateDescriptorSets(mDevice, descriptorWrites.size(),
                                   descriptorWrites.data(), 0, nullptr);
        }
    }
}
