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
    using Task = std::function<void()>;
public:
    ThreadPool(size_t n): mStop(false) {
        auto worker = [this]() {
            while (true) {
                Task task;
                {
                    std::unique_lock lock(mMutex);
                    mCondition.wait(lock, [this]() { return mStop or !mTasks.empty(); });
                    if (mStop and mTasks.empty()) {
                        return;
                    }
                    task = std::move(mTasks.front());
                    mTasks.pop();
                }
                task();
            }
        };
        for (int i = 0; i < n; ++i) {
            mThreads.emplace_back(worker);
        }
    }

    ~ThreadPool() {
        wait();
    }

    template <typename Func>
    void enqueue(Func&& task) {
        {
            std::lock_guard lock(mMutex);
            mTasks.push(std::forward<Func>(task));
        }
        mCondition.notify_one();
    }

    void wait() {
        {
            std::lock_guard lock(mMutex);
            mStop = true;
        }
        mCondition.notify_all();
        for (auto& thread: mThreads) {
            if (!thread.joinable()) continue;
            thread.join();
        }
    }

protected:

    std::vector<std::thread> mThreads;
    std::queue<Task> mTasks;

    std::mutex mMutex;
    std::condition_variable mCondition;

    bool mStop;
};


#endif //COLLECTION_THREADPOOL_HPP