#pragma once

#include <rmm/mr/device_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/cuda_stream_view.hpp>

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>

/// Lightweight RMM device_memory_resource wrapper that tracks per-task GPU
/// memory allocations.  Hot-path (alloc / dealloc) uses only atomics on
/// thread-local pointers — no locks, no maps.
class GpuMemoryTracker final : public rmm::mr::device_memory_resource {
 public:
  struct TaskMetrics {
    std::atomic<int64_t> current{0};
    std::atomic<int64_t> peak{0};

    void recordAlloc(int64_t bytes) {
      auto cur = current.fetch_add(bytes, std::memory_order_relaxed) + bytes;
      auto pk = peak.load(std::memory_order_relaxed);
      while (cur > pk &&
             !peak.compare_exchange_weak(
                 pk, cur, std::memory_order_relaxed)) {
      }
    }

    void recordDealloc(int64_t bytes) {
      current.fetch_sub(bytes, std::memory_order_relaxed);
    }
  };

  // ── Static API (called from JNI) ──────────────────────────────────────

  static void initialize() {
    if (instance_) return;
    auto* upstream = rmm::mr::get_current_device_resource();
    instance_.reset(new GpuMemoryTracker(upstream));
    rmm::mr::set_current_device_resource(instance_.get());
  }

  static void shutdown() {
    if (!instance_) return;
    rmm::mr::set_current_device_resource(instance_->upstream_);
    instance_.reset();
  }

  static void setCurrentTask(int64_t taskId) {
    auto* inst = instance_.get();
    if (!inst) return;
    auto& metrics = inst->getOrCreateTaskMetrics(taskId);
    tlTaskMetrics() = &metrics;
    tlTaskId() = taskId;
  }

  static void clearCurrentTask() {
    tlTaskMetrics() = nullptr;
    tlTaskId() = -1;
  }

  static GpuMemoryTracker* instance() { return instance_.get(); }

  static void setGpuSemaphoreMode(bool) {}

  // ── Instance API ──────────────────────────────────────────────────────

  int64_t getMaxTaskMemory(int64_t taskId) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = taskMetrics_.find(taskId);
    if (it == taskMetrics_.end()) return 0;
    return it->second->peak.load(std::memory_order_relaxed);
  }

  int64_t clearTaskMemory(int64_t taskId) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = taskMetrics_.find(taskId);
    if (it == taskMetrics_.end()) return 0;
    auto peak = it->second->peak.load(std::memory_order_relaxed);
    taskMetrics_.erase(it);
    return peak;
  }

  int64_t getTotalAllocated() const {
    return totalAllocated_.load(std::memory_order_relaxed);
  }

 private:
  explicit GpuMemoryTracker(rmm::mr::device_memory_resource* upstream)
      : upstream_(upstream) {}

  // ── device_memory_resource overrides ──────────────────────────────────

  void* do_allocate(std::size_t bytes,
                    rmm::cuda_stream_view stream) override {
    void* ptr = upstream_->allocate(stream, bytes);
    totalAllocated_.fetch_add(
        static_cast<int64_t>(bytes), std::memory_order_relaxed);
    if (auto* tm = tlTaskMetrics()) {
      tm->recordAlloc(static_cast<int64_t>(bytes));
    }
    return ptr;
  }

  void do_deallocate(void* ptr,
                     std::size_t bytes,
                     rmm::cuda_stream_view stream) noexcept override {
    upstream_->deallocate(stream, ptr, bytes);
    totalAllocated_.fetch_sub(
        static_cast<int64_t>(bytes), std::memory_order_relaxed);
    if (auto* tm = tlTaskMetrics()) {
      tm->recordDealloc(static_cast<int64_t>(bytes));
    }
  }

  [[nodiscard]] bool do_is_equal(
      device_memory_resource const& other) const noexcept override {
    return this == &other;
  }

  // ── Helpers ───────────────────────────────────────────────────────────

  TaskMetrics& getOrCreateTaskMetrics(int64_t taskId) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto& ptr = taskMetrics_[taskId];
    if (!ptr) {
      ptr = std::make_unique<TaskMetrics>();
    }
    return *ptr;
  }

  static TaskMetrics*& tlTaskMetrics() {
    thread_local TaskMetrics* tm = nullptr;
    return tm;
  }

  static int64_t& tlTaskId() {
    thread_local int64_t id = -1;
    return id;
  }

  // ── Data ──────────────────────────────────────────────────────────────

  rmm::mr::device_memory_resource* upstream_;
  std::atomic<int64_t> totalAllocated_{0};
  std::mutex mutex_;
  std::unordered_map<int64_t, std::unique_ptr<TaskMetrics>> taskMetrics_;

  static inline std::unique_ptr<GpuMemoryTracker> instance_;
};
