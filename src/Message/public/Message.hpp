//
// Created by auser on 4/4/25.
//

#ifndef VULKAN_MESSAGELOGGER_H
#define VULKAN_MESSAGELOGGER_H


#include <iostream>

namespace crv::message {
    static constexpr auto RED = "\033[31m";
    static constexpr auto GREEN = "\033[32m";
    static constexpr auto BLUE = "\033[34m";
    static constexpr auto CYAN = "\033[36m";
    static constexpr auto WHITE = "\033[37m";
    static constexpr auto GRAY = "\033[90m";
    static constexpr auto YELLOW = "\033[33m";
    static constexpr auto PINK = "\033[95m";
    static constexpr auto PURPLE = "\033[35m";

/**
* Wrapper for console output
*/

    class Message {
    public:
        const std::string reset = "\033[0m";

        Message(std::ostream &os, const std::string &color, bool autoEndOfLine = true);

        ~Message();

        Message(const Message &) = delete;

        Message &operator=(const Message &) = delete;

        Message(Message &&) = delete;

        Message &operator=(Message &&) = delete;

        template<typename T>
        Message &operator<<(T &&value) {
            mOs << mColor << std::forward<T>(value) << reset;
            return *this;
        }

    private:
        std::ostream &mOs;
        std::string mColor;
        bool mAutoEndOfLine;
    };

}

#define MESSAGE  crv::message::Message( std::cout, crv::message::GRAY   )
#define INFO     crv::message::Message( std::cout, crv::message::CYAN   )
#define DEBUG    crv::message::Message( std::cout, crv::message::BLUE   )
#define WARNING  crv::message::Message( std::cout, crv::message::YELLOW )
#define ERROR    crv::message::Message( std::cerr, crv::message::RED    )

#endif //VULKAN_MESSAGELOGGER_H
