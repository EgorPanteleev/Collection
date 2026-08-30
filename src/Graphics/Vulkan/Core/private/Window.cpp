//
// Created by igor on 4/7/26.
//

#include "Window.hpp"
#include "Message.hpp"

#include <stdexcept>

namespace crv::graphics::vulkan {
    Window::Window(const WindowCreateInfo& createInfo):
    BaseWindow(createInfo.name.c_str(), createInfo.width, createInfo.height) {
        Window::init();
    }

    void Window::init() {
        if (!glfwInit()) throw std::runtime_error("Failed to initialize GLFW");
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
        glfwWindowHint(GLFW_SCALE_TO_MONITOR, GLFW_FALSE);
        glfwWindowHint(GLFW_SRGB_CAPABLE, GLFW_TRUE);

        mWindow = glfwCreateWindow(mWidth, mHeight, mName.c_str(), nullptr, nullptr);
        if (!mWindow) {
            glfwTerminate();
            throw std::runtime_error("Failed to create GLFW window");
        }
        glfwSetInputMode(mWindow, GLFW_CURSOR, GLFW_CURSOR_HIDDEN);
    }

    std::optional<Key> Window::mapKey(const int glfwKey) {
        switch (glfwKey) {
            case GLFW_KEY_SPACE:         return Key::SPACE;
            case GLFW_KEY_APOSTROPHE:    return Key::APOSTROPHE;
            case GLFW_KEY_COMMA:         return Key::COMMA;
            case GLFW_KEY_MINUS:         return Key::MINUS;
            case GLFW_KEY_PERIOD:        return Key::PERIOD;
            case GLFW_KEY_SLASH:         return Key::SLASH;
            case GLFW_KEY_0:             return Key::NUM_0;
            case GLFW_KEY_1:             return Key::NUM_1;
            case GLFW_KEY_2:             return Key::NUM_2;
            case GLFW_KEY_3:             return Key::NUM_3;
            case GLFW_KEY_4:             return Key::NUM_4;
            case GLFW_KEY_5:             return Key::NUM_5;
            case GLFW_KEY_6:             return Key::NUM_6;
            case GLFW_KEY_7:             return Key::NUM_7;
            case GLFW_KEY_8:             return Key::NUM_8;
            case GLFW_KEY_9:             return Key::NUM_9;
            case GLFW_KEY_SEMICOLON:     return Key::SEMICOLON;
            case GLFW_KEY_EQUAL:         return Key::EQUAL;
            case GLFW_KEY_A:             return Key::A;
            case GLFW_KEY_B:             return Key::B;
            case GLFW_KEY_C:             return Key::C;
            case GLFW_KEY_D:             return Key::D;
            case GLFW_KEY_E:             return Key::E;
            case GLFW_KEY_F:             return Key::F;
            case GLFW_KEY_G:             return Key::G;
            case GLFW_KEY_H:             return Key::H;
            case GLFW_KEY_I:             return Key::I;
            case GLFW_KEY_J:             return Key::J;
            case GLFW_KEY_K:             return Key::K;
            case GLFW_KEY_L:             return Key::L;
            case GLFW_KEY_M:             return Key::M;
            case GLFW_KEY_N:             return Key::N;
            case GLFW_KEY_O:             return Key::O;
            case GLFW_KEY_P:             return Key::P;
            case GLFW_KEY_Q:             return Key::Q;
            case GLFW_KEY_R:             return Key::R;
            case GLFW_KEY_S:             return Key::S;
            case GLFW_KEY_T:             return Key::T;
            case GLFW_KEY_U:             return Key::U;
            case GLFW_KEY_V:             return Key::V;
            case GLFW_KEY_W:             return Key::W;
            case GLFW_KEY_X:             return Key::X;
            case GLFW_KEY_Y:             return Key::Y;
            case GLFW_KEY_Z:             return Key::Z;
            case GLFW_KEY_LEFT_BRACKET:  return Key::LEFT_BRACKET;
            case GLFW_KEY_BACKSLASH:     return Key::BACKSLASH;
            case GLFW_KEY_RIGHT_BRACKET: return Key::RIGHT_BRACKET;
            case GLFW_KEY_GRAVE_ACCENT:  return Key::GRAVE_ACCENT;
            case GLFW_KEY_ESCAPE:        return Key::ESCAPE;
            case GLFW_KEY_ENTER:         return Key::ENTER;
            case GLFW_KEY_TAB:           return Key::TAB;
            case GLFW_KEY_BACKSPACE:     return Key::BACKSPACE;
            case GLFW_KEY_INSERT:        return Key::INSERT;
            case GLFW_KEY_DELETE:        return Key::DELETE;
            case GLFW_KEY_RIGHT:         return Key::RIGHT;
            case GLFW_KEY_LEFT:          return Key::LEFT;
            case GLFW_KEY_DOWN:          return Key::DOWN;
            case GLFW_KEY_UP:            return Key::UP;
            case GLFW_KEY_PAGE_UP:       return Key::PAGE_UP;
            case GLFW_KEY_PAGE_DOWN:     return Key::PAGE_DOWN;
            case GLFW_KEY_HOME:          return Key::HOME;
            case GLFW_KEY_END:           return Key::END;
            case GLFW_KEY_CAPS_LOCK:     return Key::CAPS_LOCK;
            case GLFW_KEY_F1:            return Key::F1;
            case GLFW_KEY_F2:            return Key::F2;
            case GLFW_KEY_F3:            return Key::F3;
            case GLFW_KEY_F4:            return Key::F4;
            case GLFW_KEY_F5:            return Key::F5;
            case GLFW_KEY_F6:            return Key::F6;
            case GLFW_KEY_F7:            return Key::F7;
            case GLFW_KEY_F8:            return Key::F8;
            case GLFW_KEY_F9:            return Key::F9;
            case GLFW_KEY_F10:           return Key::F10;
            case GLFW_KEY_F11:           return Key::F11;
            case GLFW_KEY_F12:           return Key::F12;
            case GLFW_KEY_LEFT_SHIFT:    return Key::LEFT_SHIFT;
            case GLFW_KEY_LEFT_CONTROL:  return Key::LEFT_CONTROL;
            case GLFW_KEY_LEFT_ALT:      return Key::LEFT_ALT;
            case GLFW_KEY_LEFT_SUPER:    return Key::LEFT_SUPER;
            case GLFW_KEY_RIGHT_SHIFT:   return Key::RIGHT_SHIFT;
            case GLFW_KEY_RIGHT_CONTROL: return Key::RIGHT_CONTROL;
            case GLFW_KEY_RIGHT_ALT:     return Key::RIGHT_ALT;
            case GLFW_KEY_RIGHT_SUPER:   return Key::RIGHT_SUPER;
            default:                     return std::nullopt;
        }
    }

    std::optional<MouseButton> Window::mapMouseButton(const int glfwButton) {
        switch (glfwButton) {
            case GLFW_MOUSE_BUTTON_LEFT:   return MouseButton::LEFT;
            case GLFW_MOUSE_BUTTON_RIGHT:  return MouseButton::RIGHT;
            case GLFW_MOUSE_BUTTON_MIDDLE: return MouseButton::MIDDLE;
            case GLFW_MOUSE_BUTTON_4:      return MouseButton::BUTTON_4;
            case GLFW_MOUSE_BUTTON_5:      return MouseButton::BUTTON_5;
            case GLFW_MOUSE_BUTTON_6:      return MouseButton::BUTTON_6;
            case GLFW_MOUSE_BUTTON_7:      return MouseButton::BUTTON_7;
            case GLFW_MOUSE_BUTTON_8:      return MouseButton::BUTTON_8;
            default:                       return std::nullopt;
        }
    }
}
