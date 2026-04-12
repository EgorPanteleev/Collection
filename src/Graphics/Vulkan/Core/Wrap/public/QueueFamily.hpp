//
// Created by igor on 4/12/26.
//

#ifndef COLLECTION_QUEUEFAMILY_HPP
#define COLLECTION_QUEUEFAMILY_HPP

#include <cstdint>
#include <optional>
#include <array>

enum class QueueFamilyType {
    COMPUTE,
    GRAPHICS,
    PRESENT,
    UNKNOWN
};

struct QueueFamilyIndices {
    void set( QueueFamilyType type, uint32_t index ) { data[static_cast<int>(type)] = index; }
    [[nodiscard]] std::optional<uint32_t> get( QueueFamilyType type ) const { return data[static_cast<int>(type)]; }
    [[nodiscard]] bool isComplete() const {
        for (const auto& family: data) {
            if (!family.has_value()) return false;
        }
        return true;
    }
protected:
    std::array<std::optional<uint32_t>, static_cast<size_t>(QueueFamilyType::UNKNOWN)> data;
};

#endif //COLLECTION_QUEUEFAMILY_HPP