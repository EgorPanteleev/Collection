//
// Created by igor on 4/13/26.
//

#ifndef COLLECTION_SHADERMODULE_HPP
#define COLLECTION_SHADERMODULE_HPP

#include "DefaultWrapper.hpp"
#include "Message.hpp"

#include <vector>

namespace crv::graphics::vulkan {
    struct ShaderModuleCreateInfo{
        VkDevice device = VK_NULL_HANDLE;
        std::string fileName{};
    };

    class ShaderModule: public DefaultWrapper<VkShaderModule> {
    public:
        using DefaultWrapper::DefaultWrapper;
        explicit ShaderModule(const ShaderModuleCreateInfo& info);
        ShaderModule& operator=(ShaderModule&&) = default;
        ~ShaderModule() override { ShaderModule::destroy(); }
        void destroy() override;
    protected:
        VkDevice mDevice = VK_NULL_HANDLE;
    };
}

#endif //COLLECTION_SHADERMODULE_HPP