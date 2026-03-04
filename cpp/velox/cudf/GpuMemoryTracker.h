#pragma once

#include <rmm/mr/device_memory_resource.hpp>
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

  static void initialize();
  static void shutdown();
  static void setCurrentTask(int64_t taskId);
  static void clearCurrentTask();
  static GpuMemoryTracker* instance();
  static void setMaxConcurrentGpuTasks(int);

  // ── Instance API ──────────────────────────────────────────────────────

  int64_t getMaxTaskMemory(int64_t taskId);
  int64_t clearTaskMemory(int64_t taskId);
  int64_t getTotalAllocated() const;

 private:
  explicit GpuMemoryTracker(rmm::mr::device_memory_resource* upstream);

  void* do_allocate(std::size_t bytes,
                    rmm::cuda_stream_view stream) override;

  void do_deallocate(void* ptr,
                     std::size_t bytes,
                     rmm::cuda_stream_view stream) noexcept override;

  [[nodiscard]] bool do_is_equal(
      device_memory_resource const& other) const noexcept override;

  TaskMetrics& getOrCreateTaskMetrics(int64_t taskId);

  static TaskMetrics*& tlTaskMetrics();
  static int64_t& tlTaskId();

  rmm::mr::device_memory_resource* upstream_;
  std::atomic<int64_t> totalAllocated_{0};
  std::mutex mutex_;
  std::unordered_map<int64_t, std::unique_ptr<TaskMetrics>> taskMetrics_;

  static std::unique_ptr<GpuMemoryTracker> instance_;
  static rmm::mr::device_memory_resource* originalUpstream_;
};
