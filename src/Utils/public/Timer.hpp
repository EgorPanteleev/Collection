//
// Created by auser on 5/4/25.
//

#ifndef VULKAN_TIMER_H
#define VULKAN_TIMER_H

#include <chrono>
#include <string>

namespace crv::utils {
    class Timer {
    public:
        Timer();
        Timer(double stopTime);

        void start();

        //void stop();

        double duration();

        void setStopTime(double stopTime) { mStopTime = stopTime; }

        bool isTimeout();
    protected:
        std::chrono::high_resolution_clock::time_point mStartTime;
        double mDuration;
        double mStopTime;
    };

    class FpsCounter: public Timer {
    public:
        FpsCounter();
        void update();
        double fps() { return mFps; }
        std::string fpsAsString() { return std::to_string(mFps); }
    protected:
        uint32_t mFrames;
        double mFps;
    };
}


#endif //VULKAN_TIMER_H
