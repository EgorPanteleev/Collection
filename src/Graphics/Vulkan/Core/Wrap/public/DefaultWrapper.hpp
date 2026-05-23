//
// Created by igor on 4/11/26.
//

#ifndef COLLECTION_DEFAULTWRAPPER_HPP
#define COLLECTION_DEFAULTWRAPPER_HPP

#include <vulkan/vulkan_core.h>
#include <vector>

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
        const Type& get() const { return mHandle; }
        Type& get() { return mHandle; }
        virtual void destroy() = 0;
    protected:
        Type mHandle = VK_NULL_HANDLE;
    };

    template <typename Type>
    class VectorWrapper {
    public:
        VectorWrapper() = default;
        explicit  VectorWrapper(const std::vector<Type>& vec): mVec(vec) {}
        VectorWrapper(const VectorWrapper&) = delete;
        VectorWrapper& operator=(const VectorWrapper&) = delete;
        VectorWrapper(VectorWrapper&& other) noexcept : mVec(std::move(other.mVec)) {
            other.mVec.clear();
        }
        VectorWrapper& operator=(VectorWrapper&& other) noexcept {
            if (this != &other) {
                destroy();
                mVec = std::move(other.mVec);
                other.mVec.clear();
            }
            return *this;
        }
        virtual ~VectorWrapper() { mVec.clear(); }
        [[nodiscard]] uint32_t size() const { return mVec.size(); }
        Type get(uint32_t index) const { return mVec[index]; }
        Type& operator[](uint32_t index) { return mVec[index]; }
        const Type& operator[](uint32_t index) const { return mVec[index]; }
        virtual void destroy() = 0;
    protected:
        std::vector<Type> mVec{};
    };
}

#endif //COLLECTION_DEFAULTWRAPPER_HPP