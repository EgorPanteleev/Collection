//
// Created by igor on 4/7/26.
//

#include "BaseWindow.hpp"
#include "Message.hpp"

#include <stdexcept>

namespace crv::graphics {
    BaseWindow::BaseWindow(const char* name, const int width, const int height):
    mWidth(width), mHeight(height), mName(name) {}

    BaseWindow::BaseWindow(BaseWindow&& other) noexcept {
        mWindow = other.mWindow;
        other.mWindow = nullptr;
        mWidth = other.mWidth;
        mHeight = other.mHeight;
        mName = other.mName;
    }

    BaseWindow& BaseWindow::operator=(BaseWindow&& other) noexcept {
        if (this != &other) {
            this->~BaseWindow();
            mWindow = other.mWindow;
            other.mWindow = nullptr;
            mWidth = other.mWidth;
            mHeight = other.mHeight;
            mName = other.mName;
        }
        return *this;
    }

    BaseWindow::~BaseWindow() {
        if (!mWindow) return;
        glfwDestroyWindow(mWindow);
        glfwTerminate();
    }
}
