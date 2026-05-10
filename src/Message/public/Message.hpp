//
// Created by auser on 4/4/25.
//

#ifndef VULKAN_MESSAGELOGGER_H
#define VULKAN_MESSAGELOGGER_H


#include <iostream>

namespace crv::message {
    static constexpr auto RED        = "\033[31m";
    static constexpr auto GREEN      = "\033[32m";
    static constexpr auto BLUE       = "\033[34m";
    static constexpr auto CYAN       = "\033[36m";
    static constexpr auto WHITE      = "\033[37m";
    static constexpr auto GRAY       = "\033[90m";
    static constexpr auto LIGHT_GRAY = "\033[38;5;250m";
    static constexpr auto YELLOW     = "\033[33m";
    static constexpr auto PINK       = "\033[95m";
    static constexpr auto PURPLE     = "\033[35m";

/**
* Wrapper for console output
*/

    class Message {
    public:
        const std::string reset = "\033[0m";
        Message(std::ostream &os, const std::string& prefix, const std::string &color, bool autoEndOfLine);
        Message(std::ostream &os, const std::string &color, bool autoEndOfLine);
        ~Message();
        Message(const Message &) = delete;
        Message &operator=(const Message &) = delete;
        Message(Message &&) = delete;
        Message &operator=(Message &&) = delete;

        template<typename T>
        Message &operator<<(T &&value) {
            if (!mPrefix.empty()) {
                mOs << std::forward<T>(value);
            } else {
                mOs << mColor << std::forward<T>(value) << reset;
            }
            return *this;
        }

    private:
        std::ostream &mOs;
        std::string mColor;
        std::string mPrefix;
        bool mAutoEndOfLine;
    };

}

#define MESSAGE  crv::message::Message( std::cout,            crv::message::LIGHT_GRAY, true )
#define INFO     crv::message::Message( std::cout, "info"   , crv::message::CYAN      , true )
#define DEBUG    crv::message::Message( std::cout, "debug"  , crv::message::BLUE      , true )
#define WARNING  crv::message::Message( std::cout, "warning", crv::message::YELLOW    , true )
#define ERROR    crv::message::Message( std::cerr, "error"  , crv::message::RED       , true )

#endif //VULKAN_MESSAGELOGGER_H
