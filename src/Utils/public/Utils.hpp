//
// Created by igor on 3/10/26.
//

#ifndef COLLECTION_UTILS_HPP
#define COLLECTION_UTILS_HPP

#include <cmath>
#include <cstring>

namespace crv::utils {
    template<typename InType, typename OutType>
    inline OutType reinterpretType(const InType in) {
        OutType out;
        memcpy(&out, &in, sizeof(out));
        return out;
    }

    template <typename Type>
    inline Type addUlp(Type num, int ulps);

    template<>
    inline float addUlp(float num, int ulps) {
        if (!std::isfinite(num)) return num;
        const unsigned bits = reinterpretType<float, unsigned>(num);
        return reinterpretType<unsigned, float>(bits + ulps);
    }

    template<>
    inline double addUlp(double num, int ulps) {
        if (!std::isfinite(num)) return;
        const unsigned bits = reinterpretType<double, unsigned long long>(num);
        return reinterpretType<unsigned long long, double>(bits + ulps);
    }
}

#endif //COLLECTION_UTILS_HPP