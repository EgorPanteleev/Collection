#include <functional>
#include <thread>
#include "Vector.h"
class Sheduler {
public:
    Sheduler( int numThreads );
    template<typename Func, typename... Args>
    void addFunction(  Func func, Args&&... args ) {
        auto boundFunction = std::bind( func, std::forward<Args>(args)... );
        functions.push_back( boundFunction );
    }
    void run();
private:
    void createThread( std::function<void()> func ) const;
    int numThreads;
    Vector<std::function<void()>> functions;
    Vector<std::thread> threads;
};


