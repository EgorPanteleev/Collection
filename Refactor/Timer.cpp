//
// Created by auser on 11/26/24.
//

#include "Timer.h"

void Timer::start() {
    mStart = std::chrono::high_resolution_clock::now();
}
void Timer::end() {
    mEnd = std::chrono::high_resolution_clock::now();
}

std::string Timer::get() {
    std::chrono::duration<double> loadTime = mEnd - mStart;
    return std::to_string( loadTime.count() );
}
