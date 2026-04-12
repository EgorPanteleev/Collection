//
// Created by igor on 4/11/26.
//

#ifndef COLLECTION_DEFAULTWRAPPER_HPP
#define COLLECTION_DEFAULTWRAPPER_HPP

#include <vulkan/vulkan_core.h>

namespace crv::graphics::vulkan {
    template <typename Type>
   class DefaultWrapper {
    public:
        DefaultWrapper() = default;
        explicit  DefaultWrapper(Type handle): mHandle(handle) {}
        DefaultWrapper(const DefaultWrapper&) = delete;
        DefaultWrapper& operator=(const DefaultWrapper&) = delete;
        DefaultWrapper(DefaultWrapper&& other) noexcept : mHandle(other.mHandle) {
            other.mHandle = VK_NULL_HANDLE;
        }
        DefaultWrapper& operator=(DefaultWrapper&& other) noexcept {
            if (this != &other) {
                destroy();
                mHandle = other.mHandle;
                other.mHandle = VK_NULL_HANDLE;
            }
            return *this;
        }
        virtual ~DefaultWrapper() { mHandle = VK_NULL_HANDLE; };
        Type get() const { return mHandle; }
        virtual void destroy() = 0;
    protected:
        Type mHandle = VK_NULL_HANDLE;
    };
}

#endif //COLLECTION_DEFAULTWRAPPER_HPP