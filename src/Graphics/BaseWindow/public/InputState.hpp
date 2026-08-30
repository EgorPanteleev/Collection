//
// Created by igor on 6/12/26.
//

#ifndef COLLECTION_INPUTSTATE_HPP
#define COLLECTION_INPUTSTATE_HPP

#include <glm/vec2.hpp>

#include <array>
#include <cstdint>

namespace crv::graphics {
    enum class Key {
        SPACE, APOSTROPHE, COMMA, MINUS, PERIOD, SLASH,
        NUM_0, NUM_1, NUM_2, NUM_3, NUM_4, NUM_5, NUM_6, NUM_7, NUM_8, NUM_9,
        SEMICOLON, EQUAL,
        A, B, C, D, E, F, G, H, I, J, K, L, M,
        N, O, P, Q, R, S, T, U, V, W, X, Y, Z,
        LEFT_BRACKET, BACKSLASH, RIGHT_BRACKET, GRAVE_ACCENT,
        ESCAPE, ENTER, TAB, BACKSPACE, INSERT, DELETE,
        RIGHT, LEFT, DOWN, UP, PAGE_UP, PAGE_DOWN, HOME, END, CAPS_LOCK,
        F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, F12,
        LEFT_SHIFT, LEFT_CONTROL, LEFT_ALT, LEFT_SUPER,
        RIGHT_SHIFT, RIGHT_CONTROL, RIGHT_ALT, RIGHT_SUPER,
        COUNT
    };

    enum class MouseButton {
        LEFT, RIGHT, MIDDLE,
        BUTTON_4, BUTTON_5, BUTTON_6, BUTTON_7, BUTTON_8,
        COUNT
    };

    class InputState {
    public:
        InputState() = default;

        void beginFrame();

        void onKey(Key key, bool pressed);
        void onMouseButton(MouseButton button, bool pressed);
        void onCursorPos(double x, double y);
        void onScroll(double xOffset, double yOffset);

        [[nodiscard]] bool isPressed(Key key) const;
        [[nodiscard]] bool isReleased(Key key) const;
        [[nodiscard]] bool wasPressed(Key key) const;
        [[nodiscard]] bool wasReleased(Key key) const;

        [[nodiscard]] bool isPressed(MouseButton button) const;
        [[nodiscard]] bool isReleased(MouseButton button) const;
        [[nodiscard]] bool wasPressed(MouseButton button) const;
        [[nodiscard]] bool wasReleased(MouseButton button) const;

        [[nodiscard]] glm::dvec2 cursorPos() const { return mCursorPos; }
        [[nodiscard]] glm::dvec2 cursorDelta() const { return mCursorPos - mPrevCursorPos; }
        [[nodiscard]] glm::dvec2 scrollDelta() const { return mScrollDelta; }

    private:
        static constexpr int KEY_COUNT   = static_cast<int>(Key::COUNT);
        static constexpr int MOUSE_COUNT = static_cast<int>(MouseButton::COUNT);

        std::array<uint8_t, KEY_COUNT>   mKeys{};
        std::array<uint8_t, KEY_COUNT>   mPrevKeys{};
        std::array<uint8_t, MOUSE_COUNT> mMouse{};
        std::array<uint8_t, MOUSE_COUNT> mPrevMouse{};

        glm::dvec2 mCursorPos{0.0};
        glm::dvec2 mPrevCursorPos{0.0};
        glm::dvec2 mScrollDelta{0.0};
    };
}

#endif //COLLECTION_INPUTSTATE_HPP
