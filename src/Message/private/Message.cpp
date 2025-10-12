//
// Created by auser on 4/4/25.
//

#include "Message.hpp"

namespace crv::message {
    Message::Message(std::ostream& os, const std::string& color, bool autoEndOfLine ):
            mOs(os), mColor(color), mAutoEndOfLine(autoEndOfLine) {
    }

    Message::~Message() {
        if (mAutoEndOfLine) std::cout << std::endl;
    }
}



