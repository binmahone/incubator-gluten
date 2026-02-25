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

#include <cudf/utilities/memory_resource.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device_memory_resource.hpp>

#include <atomic>
#include <cstdint>
#include <mutex>
#include <unordered_map>

namespace gluten {

struct TaskMemoryInfo {
  int64_t currentBytes{0};
  int64_t peakBytes{0};
};

/**
 * Tracking memory resource adaptor that wraps an upstream RMM device_memory_resource
 * and tracks per-task GPU memory allocations.
 */
class GpuMemoryTracker : public rmm::mr::device_memory_resource {
 public:
  explicit GpuMemoryTracker(rmm::mr::device_memory_resource* upstream);
  ~GpuMemoryTracker() override = default;

  GpuMemoryTracker(const GpuMemoryTracker&) = delete;
  GpuMemoryTracker& operator=(const GpuMemoryTracker&) = delete;

  static void setCurrentTask(int64_t taskId);
  static void clearCurrentTask();

  int64_t getMaxTaskMemory(int64_t taskId) const;
  int64_t clearTaskMemory(int64_t taskId);
  int64_t totalAllocatedBytes() const;

  static GpuMemoryTracker* instance();
  static void initialize();
  static void shutdown();

 private:
  void* do_allocate(std::size_t bytes, rmm::cuda_stream_view stream) override;
  void do_deallocate(
      void* ptr,
      std::size_t bytes,
      rmm::cuda_stream_view stream) noexcept override;
  [[nodiscard]] bool do_is_equal(
      rmm::mr::device_memory_resource const& other) const noexcept override;

  void trackAlloc(std::size_t bytes);
  void trackDealloc(std::size_t bytes);

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
