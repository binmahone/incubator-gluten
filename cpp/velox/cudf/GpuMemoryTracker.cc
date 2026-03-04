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

// ── Static data ──────────────────────────────────────────────────────────────

std::unique_ptr<GpuMemoryTracker> GpuMemoryTracker::instance_;
rmm::mr::device_memory_resource* GpuMemoryTracker::originalUpstream_ = nullptr;

// ── Constructor ──────────────────────────────────────────────────────────────

GpuMemoryTracker::GpuMemoryTracker(rmm::mr::device_memory_resource* upstream)
    : upstream_(upstream) {}

// ── Static API ───────────────────────────────────────────────────────────────

void GpuMemoryTracker::initialize() {
  if (instance_) {
    LOG(WARNING) << "GpuMemoryTracker already initialized";
    return;
  }
  originalUpstream_ = rmm::mr::get_current_device_resource();
  if (!originalUpstream_) {
    LOG(WARNING) << "No RMM device resource found, skipping GpuMemoryTracker";
    return;
  }
  instance_.reset(new GpuMemoryTracker(originalUpstream_));
  rmm::mr::set_current_device_resource(instance_.get());
  LOG(INFO) << "GpuMemoryTracker initialized, wrapping upstream resource";
}

void GpuMemoryTracker::shutdown() {
  if (!instance_) return;
  if (originalUpstream_) {
    rmm::mr::set_current_device_resource(originalUpstream_);
  }
  instance_.reset();
  originalUpstream_ = nullptr;
  LOG(INFO) << "GpuMemoryTracker shut down";
}

void GpuMemoryTracker::setCurrentTask(int64_t taskId) {
  auto* inst = instance_.get();
  if (!inst) return;
  auto& metrics = inst->getOrCreateTaskMetrics(taskId);
  tlTaskMetrics() = &metrics;
  tlTaskId() = taskId;
}

void GpuMemoryTracker::clearCurrentTask() {
  tlTaskMetrics() = nullptr;
  tlTaskId() = -1;
}

GpuMemoryTracker* GpuMemoryTracker::instance() {
  return instance_.get();
}

void GpuMemoryTracker::setMaxConcurrentGpuTasks(int) {
  // Concurrency is managed by GpuLock, not by the memory tracker.
}

// ── Instance API ─────────────────────────────────────────────────────────────

int64_t GpuMemoryTracker::getMaxTaskMemory(int64_t taskId) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = taskMetrics_.find(taskId);
  if (it == taskMetrics_.end()) return 0;
  return it->second->peak.load(std::memory_order_relaxed);
}

int64_t GpuMemoryTracker::clearTaskMemory(int64_t taskId) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = taskMetrics_.find(taskId);
  if (it == taskMetrics_.end()) return 0;
  auto peak = it->second->peak.load(std::memory_order_relaxed);
  taskMetrics_.erase(it);
  return peak;
}

int64_t GpuMemoryTracker::getTotalAllocated() const {
  return totalAllocated_.load(std::memory_order_relaxed);
}

// ── device_memory_resource overrides ─────────────────────────────────────────

void* GpuMemoryTracker::do_allocate(
    std::size_t bytes,
    rmm::cuda_stream_view stream) {
  void* ptr = upstream_->allocate(stream, bytes);
  totalAllocated_.fetch_add(
      static_cast<int64_t>(bytes), std::memory_order_relaxed);
  if (auto* tm = tlTaskMetrics()) {
    tm->recordAlloc(static_cast<int64_t>(bytes));
  }
  return ptr;
}

void GpuMemoryTracker::do_deallocate(
    void* ptr,
    std::size_t bytes,
    rmm::cuda_stream_view stream) noexcept {
  upstream_->deallocate(stream, ptr, bytes);
  totalAllocated_.fetch_sub(
      static_cast<int64_t>(bytes), std::memory_order_relaxed);
  if (auto* tm = tlTaskMetrics()) {
    tm->recordDealloc(static_cast<int64_t>(bytes));
  }
}

bool GpuMemoryTracker::do_is_equal(
    device_memory_resource const& other) const noexcept {
  if (auto* o = dynamic_cast<const GpuMemoryTracker*>(&other)) {
    return upstream_->is_equal(*o->upstream_);
  }
  return false;
}

// ── Helpers ──────────────────────────────────────────────────────────────────

GpuMemoryTracker::TaskMetrics&
GpuMemoryTracker::getOrCreateTaskMetrics(int64_t taskId) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto& ptr = taskMetrics_[taskId];
  if (!ptr) {
    ptr = std::make_unique<TaskMetrics>();
  }
  return *ptr;
}

GpuMemoryTracker::TaskMetrics*& GpuMemoryTracker::tlTaskMetrics() {
  thread_local TaskMetrics* tm = nullptr;
  return tm;
}

int64_t& GpuMemoryTracker::tlTaskId() {
  thread_local int64_t id = -1;
  return id;
}
