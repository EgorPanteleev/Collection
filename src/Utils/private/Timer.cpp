//
// Created by auser on 5/4/25.
//

#include "Timer.hpp"

Timer::Timer(double stopTime): mStartTime(), mDuration(0), mStopTime(stopTime) {}


void Timer::start() {
    mStartTime = std::chrono::high_resolution_clock::now();
}

//void Timer::stop() {
//    auto stopTime = std::chrono::high_resolution_clock::now();
//    mDuration = std::chrono::duration<double, std::milli>( stopTime - mStartTime ).count();
//}

double Timer::duration() {
    auto currTime = std::chrono::high_resolution_clock::now();
    mDuration = std::chrono::duration<double, std::milli>( currTime - mStartTime ).count();
    return mDuration;
}

bool Timer::isTimeout() {
    return duration() >= mStopTime;
}

FpsCounter::FpsCounter(): Timer(100), mFrames(0), mFps(0) {
}

void FpsCounter::update() {
    ++mFrames;
    if (!isTimeout()) return;
    mFps = mFrames / mDuration * 1e3;
    mFrames = 0;
    start();
}
