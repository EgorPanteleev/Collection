#include "Sheduler.h"
Sheduler::Sheduler( int numThreads ): numThreads( numThreads ) {
}

void Sheduler::createThread( std::function<void()> func ) const {
    std::thread thread1( std::move( func ) );
    thread1.join();
}

void Sheduler::run() {
    size_t counter = 0;
    for ( const auto& function: functions ) {
        threads.emplace_back( std::move( function ) );
        ++counter;
        if ( counter % numThreads != 0 && functions.size() != counter ) continue;
        for ( auto& thread:threads ) {
            thread.join();
        }
        threads.clear();
    }
};