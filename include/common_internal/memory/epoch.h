#pragma once

#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <vector>

#include "common_internal/memory/thread_state.h"
#include "parlay/parallel.h"

namespace memory {

// Tracks the epoch announced by each registered thread. Each guarded
// operation announces before loading shared pointers and becomes inactive on
// exit. Reclamation uses these announcements to determine when retired
// objects are no longer reachable by readers.
class EpochManager {
 public:
  static constexpr long kInactiveEpoch = std::numeric_limits<long>::max();
  using GuardSlot = ThreadState*;

  static EpochManager& Get() {
    // Thread-local registrations can outlive other static objects during
    // process teardown, so the process-lifetime manager is not destroyed.
    static EpochManager* instance = new EpochManager();
    return *instance;
  }

  long GetCurrentEpoch() const {
    return current_epoch_.load(std::memory_order_relaxed);
  }

  GuardSlot RegisterGuardSlot() {
    std::lock_guard<std::mutex> lock(slot_registration_mutex_);
    size_t id;
    if (free_guard_slots_.empty()) {
      id = next_guard_slot_++;
      if (id >= num_workers_) std::abort();
    } else {
      id = free_guard_slots_.back();
      free_guard_slots_.pop_back();
    }
    slots_[id].announced_epoch.store(
        kInactiveEpoch, std::memory_order_relaxed);
    slots_[id].depth = 0;
    return &slots_[id];
  }

  void UnregisterGuardSlot(GuardSlot raw_slot) {
    auto* slot = raw_slot;
    slot->announced_epoch.store(kInactiveEpoch, std::memory_order_release);
    slot->depth = 0;
    const size_t id = static_cast<size_t>(slot - slots_.data());
    assert(id < num_workers_);
    std::lock_guard<std::mutex> lock(slot_registration_mutex_);
    free_guard_slots_.push_back(id);
  }

  void EnterSlot(GuardSlot raw_slot) {
    auto& slot = *raw_slot;
    while (true) {
      long epoch = current_epoch_.load(std::memory_order_acquire);
      slot.announced_epoch.store(epoch, std::memory_order_release);
      if (current_epoch_.load(std::memory_order_acquire) == epoch) return;
    }
  }

  static void ExitSlot(GuardSlot raw_slot) {
    auto& slot = *raw_slot;
    slot.announced_epoch.store(kInactiveEpoch, std::memory_order_release);
  }

  void EnterOperationSlot(GuardSlot raw_slot) {
    auto& slot = *raw_slot;
    assert(slot.depth == 0 &&
           slot.announced_epoch.load(std::memory_order_relaxed) ==
               kInactiveEpoch &&
           "OperationEpochGuard must not be nested");
    EnterSlot(raw_slot);
  }

  static void ExitOperationSlot(GuardSlot raw_slot) {
    auto& slot = *raw_slot;
    assert(slot.depth == 0 &&
           "OperationEpochGuard must not contain a nested EpochGuard");
    ExitSlot(raw_slot);
  }

  void EnterNestedSlot(GuardSlot raw_slot) {
    auto& slot = *raw_slot;
    if (slot.depth++ != 0) return;
    EnterSlot(raw_slot);
  }

  static void ExitNestedSlot(GuardSlot raw_slot) {
    auto& slot = *raw_slot;
    assert(slot.depth > 0);
    if (--slot.depth == 0) ExitSlot(raw_slot);
  }

  // Only one thread scans at a time. Advancing from e to e+1 is safe when
  // every active thread either is quiescent or has announced e. A thread that
  // remains in e then prevents the next advance until it exits.
  bool TryAdvanceEpoch() {
    if (advance_lock_.test_and_set(std::memory_order_acquire)) return false;

    long current = current_epoch_.load(std::memory_order_acquire);
    bool can_advance = true;
    for (size_t i = 0; i < num_workers_; i++) {
      long announced =
          slots_[i].announced_epoch.load(std::memory_order_acquire);
      if (announced != kInactiveEpoch && announced != current) {
        can_advance = false;
        break;
      }
    }

    if (can_advance) {
      current_epoch_.store(current + 1, std::memory_order_release);
    }
    advance_lock_.clear(std::memory_order_release);
    return can_advance;
  }

  size_t GetNumWorkers() const { return num_workers_; }

  ThreadState& GetThreadState(size_t id) {
    assert(id < num_workers_);
    return slots_[id];
  }

  void Reset() {
    current_epoch_.store(0, std::memory_order_release);
    for (size_t i = 0; i < num_workers_; i++) {
      slots_[i].announced_epoch.store(
          kInactiveEpoch, std::memory_order_release);
      slots_[i].depth = 0;
    }
    advance_lock_.clear(std::memory_order_release);
  }

  long GetAnnouncedEpoch(GuardSlot raw_slot) const {
    const auto& slot = *raw_slot;
    return slot.announced_epoch.load(std::memory_order_acquire);
  }

  // Retire normally runs inside an epoch guard. Reuse the thread's private
  // announcement instead of reading the shared global epoch on every update.
  // Unguarded test and teardown paths fall back to the global value.
  long GetRetireEpoch(GuardSlot raw_slot) const {
    const auto& slot = *raw_slot;
    const long announced =
        slot.announced_epoch.load(std::memory_order_relaxed);
    if (announced != kInactiveEpoch) return announced;
    return current_epoch_.load(std::memory_order_acquire);
  }

 private:
  EpochManager() : num_workers_(parlay::num_workers()), slots_(num_workers_) {
    assert(reinterpret_cast<uintptr_t>(slots_.data()) % 64 == 0 &&
           "ThreadState array must be 64-byte aligned to avoid false sharing");
  }

  size_t num_workers_;
  std::atomic<long> current_epoch_{0};
  std::atomic_flag advance_lock_ = ATOMIC_FLAG_INIT;
  std::mutex slot_registration_mutex_;
  size_t next_guard_slot_{0};
  std::vector<size_t> free_guard_slots_;
  std::vector<ThreadState> slots_;
};

// Each OS thread registers one stable epoch slot. This avoids querying the TBB
// arena for a transient worker index on every index operation.
class ThreadEpochRegistration {
 public:
  ThreadEpochRegistration()
      : manager_(&EpochManager::Get()), slot_(manager_->RegisterGuardSlot()) {}

  ~ThreadEpochRegistration() { manager_->UnregisterGuardSlot(slot_); }

  EpochManager* manager() const { return manager_; }
  EpochManager::GuardSlot slot() const { return slot_; }

  ThreadEpochRegistration(const ThreadEpochRegistration&) = delete;
  ThreadEpochRegistration& operator=(const ThreadEpochRegistration&) = delete;

 private:
  EpochManager* manager_;
  EpochManager::GuardSlot slot_;
};

inline ThreadEpochRegistration& GetThreadEpochRegistration() {
  static thread_local ThreadEpochRegistration registration;
  return registration;
}

// General guard for the rare call paths that may nest.
class EpochGuard {
 public:
  EpochGuard() {
#ifndef FH_INDEX_DISABLE_RECLAMATION
    auto& registration = GetThreadEpochRegistration();
    slot_ = registration.slot();
    registration.manager()->EnterNestedSlot(slot_);
#endif
  }

  ~EpochGuard() {
#ifndef FH_INDEX_DISABLE_RECLAMATION
    EpochManager::ExitNestedSlot(slot_);
#endif
  }

  EpochGuard(const EpochGuard&) = delete;
  EpochGuard& operator=(const EpochGuard&) = delete;
  EpochGuard(EpochGuard&&) = delete;
  EpochGuard& operator=(EpochGuard&&) = delete;

 private:
#ifndef FH_INDEX_DISABLE_RECLAMATION
  EpochManager::GuardSlot slot_;
#endif
};

// FH-index public operations do not nest guards. Their hot path can therefore
// announce directly without maintaining a nesting depth on every operation.
class OperationEpochGuard {
 public:
  OperationEpochGuard() {
#ifndef FH_INDEX_DISABLE_RECLAMATION
    auto& registration = GetThreadEpochRegistration();
    slot_ = registration.slot();
    registration.manager()->EnterOperationSlot(slot_);
#endif
  }

  ~OperationEpochGuard() {
#ifndef FH_INDEX_DISABLE_RECLAMATION
    EpochManager::ExitOperationSlot(slot_);
#endif
  }

  OperationEpochGuard(const OperationEpochGuard&) = delete;
  OperationEpochGuard& operator=(const OperationEpochGuard&) = delete;
  OperationEpochGuard(OperationEpochGuard&&) = delete;
  OperationEpochGuard& operator=(OperationEpochGuard&&) = delete;

 private:
#ifndef FH_INDEX_DISABLE_RECLAMATION
  EpochManager::GuardSlot slot_;
#endif
};

}  // namespace memory
