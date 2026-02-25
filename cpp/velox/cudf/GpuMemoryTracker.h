/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <rmm/mr/device_memory_resource.hpp>
#include <rmm/cuda_stream_view.hpp>

#include <atomic>
#include <cstdint>
#include <mutex>
#include <unordered_map>

namespace gluten {

/// Per-task memory tracking data (accessed under tasksMutex_).
struct TaskMemoryInfo {
  int64_t currentBytes{0};
  int64_t peakBytes{0};
};

/**
 * A tracking memory resource adaptor that wraps an existing RMM device_memory_resource and tracks
 * per-task GPU memory allocations.
 *
 * Each Spark task thread calls setCurrentTask() before GPU work and clearCurrentTask() after.
 * All RMM allocations/deallocations on that thread are attributed to the task.
 *
 * Thread safety: the tracker uses a mutex to protect the per-task map. The thread-local task ID
 * is lock-free. allocate/deallocate contention is minimal since updates to different task entries
 * are independent.
 */
class GpuMemoryTracker : public rmm::mr::device_memory_resource {
 public:
  explicit GpuMemoryTracker(rmm::mr::device_memory_resource* upstream);
  ~GpuMemoryTracker() override = default;

  GpuMemoryTracker(const GpuMemoryTracker&) = delete;
  GpuMemoryTracker& operator=(const GpuMemoryTracker&) = delete;

  /// Set the task ID for the calling thread. Must be called before GPU work.
  static void setCurrentTask(int64_t taskId);

  /// Clear the task ID for the calling thread.
  static void clearCurrentTask();

  /// Get the peak GPU memory usage for the given task (bytes). Returns 0 if unknown.
  int64_t getMaxTaskMemory(int64_t taskId) const;

  /// Remove tracking data for a completed task. Returns the peak memory before removal.
  int64_t clearTaskMemory(int64_t taskId);

  /// Get total GPU memory currently allocated across all tasks.
  int64_t totalAllocatedBytes() const;

  /// Get the singleton instance (nullptr if not initialized).
  static GpuMemoryTracker* instance();

  /// Initialize the global tracker by wrapping the current RMM device resource.
  static void initialize();

  /// Shutdown: restore original RMM resource and destroy the tracker.
  static void shutdown();

 protected:
  void* do_allocate(std::size_t bytes, rmm::cuda_stream_view stream) override;
  void do_deallocate(
      void* ptr,
      std::size_t bytes,
      rmm::cuda_stream_view stream) noexcept override;
  [[nodiscard]] bool do_is_equal(
      device_memory_resource const& other) const noexcept override;

 private:
  rmm::mr::device_memory_resource* upstream_;
  mutable std::mutex tasksMutex_;
  std::unordered_map<int64_t, TaskMemoryInfo> tasks_;
  std::atomic<int64_t> totalAllocated_{0};

  static thread_local int64_t currentTaskId_;
  static thread_local bool hasTask_;
  static std::unique_ptr<GpuMemoryTracker> instance_;
  static rmm::mr::device_memory_resource* originalResource_;
};

} // namespace gluten
