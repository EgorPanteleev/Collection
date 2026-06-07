//
// Created by igor on 3/23/26.
//

#ifndef COLLECTION_THREADPOOL_HPP
#define COLLECTION_THREADPOOL_HPP

#include <vector>
#include <queue>
#include <thread>
#include <functional>

#include <mutex>
#include <condition_variable>

class ThreadPool {
public:
    using Task = std::function<void()>;
    explicit ThreadPool(const uint32_t numWorkers) {
        auto worker = [this] {
            while (true) {
                Task task{};
                {
                    std::unique_lock lock{mMutex};
                    mConditionVariable.wait(lock, [this]{ return !mTasks.empty() or mStop; });
                    if (mStop and mTasks.empty()) return;
                    task = std::move(mTasks.front());
                    mTasks.pop();
                }
                task();
            }
        };
        for (uint32_t i = 0; i < numWorkers; ++i) {
            mWorkers.emplace_back(worker);
        }
    }

    ~ThreadPool() {
        stop();
    }

    template<typename Func>
    void enqueue(Func&& task) {
        {
            std::lock_guard lock{mMutex};
            mTasks.emplace(std::forward<Func>(task));
        }
        mConditionVariable.notify_one();
    }

    void stop() {
        {
            std::lock_guard lock{mMutex};
            mStop = true;
        }
        mConditionVariable.notify_all();
        for (auto& worker: mWorkers) {
            if (!worker.joinable()) continue;
            worker.join();
        }
    }
private:
    std::queue<Task> mTasks{};
    std::vector<std::thread> mWorkers{};
    std::condition_variable mConditionVariable{};
    std::mutex mMutex{};
    bool mStop = false;
};


#endif //COLLECTION_THREADPOOL_HPP