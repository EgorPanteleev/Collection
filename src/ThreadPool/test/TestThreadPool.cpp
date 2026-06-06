//
// Created by igor on 3/23/26.
//

#include <gtest/gtest.h>
#include <random>

#include "ThreadPool.hpp"
#include "Message.hpp"

std::vector<float> generateNumbers(size_t n) {
    std::vector<float> res;
    std::mt19937 gen(41);
    std::uniform_int_distribution dist(-1000, 1000);

    for (size_t i = 0; i < n; ++i) {
        res.emplace_back(dist(gen));
    }
    return res;
}

TEST(ThreadPool, All) {
    constexpr int n = 100;
    ThreadPool pool(n);
    auto vec = generateNumbers(n);
    float sum = std::reduce(vec.begin(), vec.end());
    std::atomic<float> calc = 0;
    for (int i = 0; i < n; ++i) {
        pool.enqueue([&vec, &calc, i](){ calc.fetch_add(vec[i]); });
    }
    pool.stop();
    EXPECT_EQ(sum, calc);
}