#include <iostream>
#include <chrono>
#include <thread>
#include <vector>
#include <atomic>

class HighResolutionTimer {
public:
    HighResolutionTimer(size_t worker_count) {
        thread_data.resize(worker_count);
    }

    void reset() {
        for (auto& data : thread_data) {
            if (data == nullptr) {
                continue;
            }
            data->total_duration = 0;
            data->call_count = 0;
        }
    }

    void start_timing(size_t tid) {
        assert(tid < thread_data.size());
        auto data = get_thread_local_data(tid);
        data->start_time = std::chrono::high_resolution_clock::now();
    }

    void end_timing(size_t tid) {
        assert(tid < thread_data.size());
        auto data = get_thread_local_data(tid);
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - data->start_time).count();

        data->total_duration += duration;
        data->call_count++;
    }

    long long get_total_duration_ns() {
        long long total = 0;
        for (auto& data : thread_data) {
            if (data == nullptr) {
                continue;
            }
            total += data->total_duration.load();
        }
        return total;
    }

    long long get_call_count() {
        long long total = 0;
        for (auto& data : thread_data) {
            if (data == nullptr) {
                continue;
            }
            total += data->call_count.load();
        }
        return total;
    }

    // 每个线程的局部数据
    struct alignas(64) ThreadLocalData {
        std::chrono::high_resolution_clock::time_point start_time;
        std::atomic<long long> total_duration{0}; // 当前线程的总时间（纳秒）
        std::atomic<long long> call_count{0};    // 当前线程的调用次数
    };

    // 获取当前线程的局部数据
    ThreadLocalData* get_thread_local_data(size_t tid) {
        assert(tid < thread_data.size());
        ThreadLocalData* data = thread_data[tid].get();
        if (data == nullptr) {
            thread_data[tid] = std::make_unique<ThreadLocalData>();
            data = thread_data[tid].get();
        }
        return data;
    }

    std::vector<std::unique_ptr<ThreadLocalData>> thread_data; // 存储所有线程的局部数据 shared_ptr
};