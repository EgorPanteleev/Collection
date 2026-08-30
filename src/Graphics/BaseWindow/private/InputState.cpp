//
// Created by igor on 6/12/26.
//

#include "InputState.hpp"

namespace crv::graphics {
    void InputState::beginFrame() {
        mPrevKeys = mKeys;
        mPrevMouse = mMouse;
        mPrevCursorPos = mCursorPos;
        mScrollDelta = glm::dvec2(0.0);
    }

    void InputState::onKey(Key key, bool pressed) {
        mKeys[static_cast<int>(key)] = pressed ? 1 : 0;
    }

    void InputState::onMouseButton(MouseButton button, bool pressed) {
        mMouse[static_cast<int>(button)] = pressed ? 1 : 0;
    }

    void InputState::onCursorPos(double x, double y) {
        mCursorPos = {x, y};
    }

    void InputState::onScroll(double xOffset, double yOffset) {
        mScrollDelta += glm::dvec2(xOffset, yOffset);
    }

    bool InputState::isPressed(Key key) const {
        return mKeys[static_cast<int>(key)] != 0;
    }

    bool InputState::isReleased(Key key) const {
        return mKeys[static_cast<int>(key)] == 0;
    }

    bool InputState::wasPressed(Key key) const {
        const int index = static_cast<int>(key);
        return mKeys[index] != 0 && mPrevKeys[index] == 0;
    }

    bool InputState::wasReleased(Key key) const {
        const int index = static_cast<int>(key);
        return mKeys[index] == 0 && mPrevKeys[index] != 0;
    }

    bool InputState::isPressed(MouseButton button) const {
        return mMouse[static_cast<int>(button)] != 0;
    }

    bool InputState::isReleased(MouseButton button) const {
        return mMouse[static_cast<int>(button)] == 0;
    }

    bool InputState::wasPressed(MouseButton button) const {
        const int index = static_cast<int>(button);
        return mMouse[index] != 0 && mPrevMouse[index] == 0;
    }

    bool InputState::wasReleased(MouseButton button) const {
        const int index = static_cast<int>(button);
        return mMouse[index] == 0 && mPrevMouse[index] != 0;
    }
}
