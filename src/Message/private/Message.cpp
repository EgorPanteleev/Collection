//
// Created by auser on 4/4/25.
//

#include "Message.hpp"

namespace crv::message {
    Message::Message(std::ostream &os, const std::string &prefix, const std::string &color, bool autoEndOfLine):
    mOs(os), mColor(color), mPrefix(prefix), mAutoEndOfLine(autoEndOfLine) {
        if (!mPrefix.empty()) mOs << "[" << mColor << mPrefix << reset << "] ";
    }

    Message::Message(std::ostream &os, const std::string &color, bool autoEndOfLine):
    Message(os, color, "", autoEndOfLine)  {}

    Message::~Message() {
        if (mAutoEndOfLine) std::cout << std::endl;
    }
}



