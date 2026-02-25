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

#include "GpuMemoryTracker.h"

#include <rmm/mr/per_device_resource.hpp>
#include <glog/logging.h>

#include <algorithm>

namespace gluten {

thread_local int64_t GpuMemoryTracker::currentTaskId_ = -1;
thread_local bool GpuMemoryTracker::hasTask_ = false;
std::unique_ptr<GpuMemoryTracker> GpuMemoryTracker::instance_ = nullptr;
rmm::mr::device_memory_resource* GpuMemoryTracker::originalResource_ = nullptr;

GpuMemoryTracker::GpuMemoryTracker(rmm::mr::device_memory_resource* upstream)
    : upstream_(upstream) {}

void GpuMemoryTracker::setCurrentTask(int64_t taskId) {
  currentTaskId_ = taskId;
  hasTask_ = true;
}

void GpuMemoryTracker::clearCurrentTask() {
  hasTask_ = false;
  currentTaskId_ = -1;
}

int64_t GpuMemoryTracker::getMaxTaskMemory(int64_t taskId) const {
  std::lock_guard<std::mutex> lock(tasksMutex_);
  auto it = tasks_.find(taskId);
  if (it != tasks_.end()) {
    return it->second.peakBytes;
  }
  return 0;
}

int64_t GpuMemoryTracker::clearTaskMemory(int64_t taskId) {
  std::lock_guard<std::mutex> lock(tasksMutex_);
  auto it = tasks_.find(taskId);
  if (it != tasks_.end()) {
    int64_t peak = it->second.peakBytes;
    tasks_.erase(it);
    return peak;
  }
  return 0;
}

int64_t GpuMemoryTracker::totalAllocatedBytes() const {
  return totalAllocated_.load(std::memory_order_relaxed);
}

GpuMemoryTracker* GpuMemoryTracker::instance() {
  return instance_.get();
}

void GpuMemoryTracker::initialize() {
  if (instance_) {
    LOG(WARNING) << "GpuMemoryTracker already initialized";
    return;
  }
  originalResource_ = rmm::mr::get_current_device_resource();
  if (!originalResource_) {
    LOG(WARNING) << "No RMM device resource found, skipping GpuMemoryTracker";
    return;
  }
  instance_ = std::make_unique<GpuMemoryTracker>(originalResource_);
  rmm::mr::set_current_device_resource(instance_.get());
  LOG(INFO) << "GpuMemoryTracker initialized, wrapping upstream resource";
}

void GpuMemoryTracker::shutdown() {
  if (!instance_) {
    return;
  }
  if (originalResource_) {
    rmm::mr::set_current_device_resource(originalResource_);
  }
  instance_.reset();
  originalResource_ = nullptr;
  LOG(INFO) << "GpuMemoryTracker shut down";
}

void* GpuMemoryTracker::do_allocate(
    std::size_t bytes,
    rmm::cuda_stream_view stream) {
  void* ptr = upstream_->allocate(stream, bytes);
  if (hasTask_) {
    int64_t signedBytes = static_cast<int64_t>(bytes);
    totalAllocated_.fetch_add(signedBytes, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lock(tasksMutex_);
    auto& info = tasks_[currentTaskId_];
    info.currentBytes += signedBytes;
    info.peakBytes = std::max(info.peakBytes, info.currentBytes);
  }
  return ptr;
}

void GpuMemoryTracker::do_deallocate(
    void* ptr,
    std::size_t bytes,
    rmm::cuda_stream_view stream) noexcept {
  upstream_->deallocate(stream, ptr, bytes);
  if (hasTask_) {
    int64_t signedBytes = static_cast<int64_t>(bytes);
    totalAllocated_.fetch_sub(signedBytes, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lock(tasksMutex_);
    auto it = tasks_.find(currentTaskId_);
    if (it != tasks_.end()) {
      it->second.currentBytes -= signedBytes;
    }
  }
}

bool GpuMemoryTracker::do_is_equal(
    device_memory_resource const& other) const noexcept {
  if (auto* o = dynamic_cast<const GpuMemoryTracker*>(&other)) {
    return upstream_->is_equal(*o->upstream_);
  }
  return false;
}

} // namespace gluten
