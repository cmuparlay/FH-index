#pragma once
#include <ctime>
#include <cstdio>
#include <string>
#include <iostream>
#include <chrono>
#include <set>
#include <map>
#include <vector>
#include <cassert>
#include <iomanip>
#include <mutex>
#include <atomic>
#include <thread>

enum print_type { pt_full, pt_time, pt_name };

class timer {

   public:
    inline static bool active = false;
    inline static bool default_detail = false;
    inline static bool print_when_time = false;
    inline static timer* root_timer = nullptr;

    inline static thread_local timer* current_timer = nullptr;

    std::string name;
    timer* parent;
    std::thread::id tid;
    std::mutex* mut;

    std::string name_with_prefix;
    std::chrono::duration<double> total_time;
    int count;
    std::chrono::high_resolution_clock::time_point start_time, end_time;
    std::vector<double> details;
    std::map<std::string, timer*> sub_timers;

    timer(std::string _name, timer* _parent) {
        name = _name;
        parent = _parent;
        tid = std::this_thread::get_id();
        mut = nullptr;

        if (parent != nullptr) {
            name_with_prefix = parent->name_with_prefix + " -> " + name;
        } else {
            name_with_prefix = name;
        }

        total_time = std::chrono::duration<double>();
        count = 0;
        details.clear();
        sub_timers.clear();
    }

    ~timer() {
        if (mut) {
            mut->lock();
        }
        for (auto& sub_timer_pairs : sub_timers) {
            delete sub_timer_pairs.second;
        }
        sub_timers.clear();
        if (mut) {
            mut->unlock();
        }
    }

    // should not be called when any timer is active
    void print(print_type pt) {
        if (count == 0) return;
        assert(active == false);

        if (pt == pt_name) {
            std::cout << name_with_prefix << std::endl;
        } else if (pt == pt_time) {
            printf("Average Time: %lf\n", total_time.count() / this->count);
        } else {
            std::cout << "/----------------------------------------\\" << std::endl;
            std::cout << "Timer: " << name_with_prefix << ": " << std::endl;
            std::cout << std::setw(20) << "Average Time"
                 << " : " << total_time.count() / this->count << std::endl;
            std::cout << std::setw(20) << "Total Time"
                 << " : " << total_time.count() << std::endl;
            printf("Proportion: \n");
            double total_proportion = 1.0;
            for (auto& sub_timer_pair : sub_timers) {
                auto sub_timer = sub_timer_pair.second;
                double sub_time = sub_timer->total_time.count();
                double sub_average_time = sub_time / sub_timer->count;
                double proportion = sub_time / total_time.count();
                total_proportion -= proportion;
                int proportion_int = (int)(proportion * 100.0);

                std::string sub_name = sub_timer_pair.second->name;
                std::cout << std::left << " -> " << std::setw(16) << sub_name
                     << " : " << std::setw(3) << proportion_int << "%" << " : " << sub_average_time << std::endl;
            }
            total_proportion = std::max(0.0, total_proportion);
            int total_proportion_int = (int)(total_proportion * 100.0);
            std::cout << std::left << " -> " << std::setw(16) << "other"
                 << " : " << std::setw(3) << total_proportion_int << "%" << std::endl;
            printf("Details: ");
            for (size_t i = 0; i < details.size(); i++) {
                printf("%lf ", details[i]);
            }
            std::cout << std::endl;
            printf("Occurance: %d\n", count);
            std::cout << "\\----------------------------------------/" << std::endl << std::endl;
        }
        fflush(stdout);
    }

    void start() { 
        assert(tid == std::this_thread::get_id());
        start_time = std::chrono::high_resolution_clock::now();
    }
    void end(bool detail) {
        assert(tid == std::this_thread::get_id());
        if (active) {
            end_time = std::chrono::high_resolution_clock::now();
            auto d = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time);
            total_time += d;
            count++;
            if (detail) {
                details.push_back(d.count());
            }
        }
    }
    void reset() {
        total_time = std::chrono::duration<double>();
        count = 0;
        details.clear();
    }
};


inline timer* get_root_timer() {
    assert(timer::root_timer != nullptr);
    return timer::root_timer;
}

inline void init_root_timer() {
    assert(timer::root_timer == nullptr);
    timer* rt = new timer("root", nullptr);
    timer::root_timer = rt;
}

template<bool concurrent = false>
inline timer* time_start(std::string name, timer* parent) {
    timer* previous_timer = nullptr;
    if (parent != nullptr) {
        previous_timer = parent;
    } else {
        if (timer::current_timer == nullptr) {
            timer::current_timer = get_root_timer();
        }
        previous_timer = timer::current_timer;
    }

    if (concurrent) {
        if (!previous_timer->mut) {
            previous_timer->mut = new std::mutex();
        }
        previous_timer->mut->lock();
    }
    if (!previous_timer->sub_timers.count(name)) {
        timer* tt = new timer(name, previous_timer);
        previous_timer->sub_timers[name] = tt;
    }
    timer* t = previous_timer->sub_timers[name];
    if (concurrent) {
        previous_timer->mut->unlock();
    }

    if (timer::print_when_time) {
        printf("%s\n", t->name_with_prefix.c_str());
        // std::cout<<t->name_with_prefix<<std::endl;
        // t->print(print_type::pt_name);
    }
    t->start();
    timer::current_timer = t;
    return t;
}

template<bool concurrent = false>
inline void time_end(std::string name, bool detail = timer::default_detail) {
    assert(timer::current_timer != nullptr);
    assert(timer::current_timer != get_root_timer());
    timer* t = timer::current_timer;

#ifndef NDEBUG
    if (concurrent) {
        assert(t->parent->mut);
        t->parent->mut->lock();
    }
    assert(t == t->parent->sub_timers[name]);
    if (concurrent) {
        t->parent->mut->unlock();
    }
#endif

    t->end(detail);
    timer::current_timer = t->parent;
    assert(timer::current_timer != nullptr);
}

template <bool concurrent = false, class F>
inline void time_nested(std::string name, F f, timer* parent = nullptr, bool detail = timer::default_detail) {
    time_start<concurrent>(name, parent);
    if (!concurrent && timer::current_timer->parent != nullptr) {
        assert(timer::current_timer->parent->tid == std::this_thread::get_id());
    }
    f();
    time_end<concurrent>(name, detail);
}

template<bool concurrent = false, class F>
inline void time_nested_pass(std::string name, F f, timer* parent = nullptr, bool detail = timer::default_detail) {
    timer* timer = time_start<concurrent>(name, parent);
    f(timer);
    time_end<concurrent>(name, detail);
}

template <class F>
inline void apply_to_timer_tree(timer* t, F f) {
    assert(timer::active == false);
    f(t);
    for (auto& timer : t->sub_timers) {
        apply_to_timer_tree(timer.second, f);
    }
}

template <class F>
inline void apply_to_all_timers_recursive(F f) {
    apply_to_timer_tree(get_root_timer(), f);
}

inline void print_all_timers(print_type pt) {
    apply_to_all_timers_recursive([&](timer* t) { t->print(pt); });
}

inline void reset_all_timers() {
    apply_to_all_timers_recursive([&](timer* t) { t->reset(); });
}

class coverage_timer {
    std::chrono::high_resolution_clock::time_point last_active_time, last_inactive_time;
    double active_time, inactive_time;
    int count;
    bool init_state;
    std::mutex mut;
    std::string name;
    // std::vector<double> active_times, inactive_times;

   public:
    coverage_timer(std::string _name) {
        name = _name;
        reset();
    }

    void start() {
        std::unique_lock wLock(mut);
        if (count == 0) {
            auto current_time = std::chrono::high_resolution_clock::now();
            if (!init_state) {
                double d = std::chrono::duration_cast<std::chrono::duration<double>>(current_time -
                                                         last_inactive_time)
                             .count();
                inactive_time += d;
                // inactive_times.push_back(d);
            }
            last_active_time = current_time;
        }
        init_state = false;
        count++;
    }

    void end() {
        std::unique_lock wLock(mut);
        assert(count > 0);
        count--;
        if (count == 0) {
            auto current_time = std::chrono::high_resolution_clock::now();
            double d =
                std::chrono::duration_cast<std::chrono::duration<double>>(current_time - last_active_time)
                    .count();
            active_time += d;
            // active_times.push_back(d);
            last_inactive_time = current_time;
        }
    }

    void print_vector(std::vector<double>& x) {
        for (size_t i = 0; i < x.size(); i ++) {
            printf("%d %lf\n", i, x[i]);
        }
    }

    void print(print_type pt) {
        std::unique_lock wLock(mut);
        assert(count == 0);
        if (pt == pt_name) {
            std::cout << name << std::endl;
        } else {
            printf("%s:\n", name.c_str());
            printf("Active Time: %lf\n", active_time);
            printf("Inactive Time: %lf\n", inactive_time);
            printf("Active Ratio: %lf\n",
                   active_time / (active_time + inactive_time));
            // printf("Active Times:\n");
            // print_vector(active_times);
            // printf("Inactive Times:\n");
            // print_vector(inactive_times);
        }
    }

    void reset() {
        active_time = inactive_time = 0;
        count = 0;
        init_state = true;
        // active_times.clear();
        // inactive_times.clear();
    }
};

coverage_timer* cpu_coverage_timer = new coverage_timer("CPU coverage timer");
coverage_timer* pim_coverage_timer = new coverage_timer("PIM coverage timer");