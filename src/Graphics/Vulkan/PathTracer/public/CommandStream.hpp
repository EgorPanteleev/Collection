//
// Created by igor on 6/12/26.
//

#ifndef COLLECTION_COMMANDSTREAM_HPP
#define COLLECTION_COMMANDSTREAM_HPP

#include "Command.hpp"

#include <vector>

namespace crv::graphics::vulkan {
    class CommandStream {
    public:
        void push(const Command command) { mCommands.push_back(command); }
        void push(const CommandType type) { mCommands.push_back(Command{type}); }
        void clear() { mCommands.clear(); }
        [[nodiscard]] bool empty() const { return mCommands.empty(); }
        [[nodiscard]] const std::vector<Command>& get() const { return mCommands; }
    private:
        std::vector<Command> mCommands{};
    };
}

#endif //COLLECTION_COMMANDSTREAM_HPP
