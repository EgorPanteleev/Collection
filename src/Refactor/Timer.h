//
// Created by auser on 11/26/24.
//

#ifndef COLLECTION_TIMER_H
#define COLLECTION_TIMER_H
#include <chrono>
#include <string>

class Timer {
    using timePoint = std::chrono::_V2::system_clock::time_point;
public:
    void start();
    void end();
    std::string get();
private:
    timePoint mStart;
    timePoint mEnd;
};


#endif //COLLECTION_TIMER_H
