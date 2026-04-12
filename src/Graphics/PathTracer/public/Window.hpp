//
// Created by igor on 10/17/25.
//

#ifndef WINDOW_HPP
#define WINDOW_HPP

#include "BaseWindow.hpp"

namespace crv::graphics {
    class Window: public BaseWindow {
    public:
        using BaseWindow::BaseWindow;
        ~Window() override = default;
        void init() override;
    };
}

#endif //WINDOW_HPP
