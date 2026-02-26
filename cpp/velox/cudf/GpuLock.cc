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

#include "GpuLock.h"
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <optional>
#include <glog/logging.h>

namespace gluten {

namespace {

std::atomic<bool> gSemaphoreMode{false};

struct GpuLockState {
  std::mutex gGpuMutex;
  std::condition_variable gGpuCv;
  std::optional<std::thread::id> gGpuOwner;
};

GpuLockState& getGpuLockState() {
  static GpuLockState state;
  return state;
}

} // namespace

void setGpuSemaphoreMode(bool enabled) {
  gSemaphoreMode.store(enabled, std::memory_order_release);
  if (enabled) {
    LOG(INFO) << "GpuLock disabled: concurrency managed by Java GpuSemaphore";
  }
}

bool isGpuSemaphoreMode() {
  return gSemaphoreMode.load(std::memory_order_acquire);
}

void lockGpu() {
  if (gSemaphoreMode.load(std::memory_order_acquire)) {
    return;
  }
  std::thread::id tid = std::this_thread::get_id();
  std::unique_lock<std::mutex> lock(getGpuLockState().gGpuMutex);
  if (getGpuLockState().gGpuOwner == tid) {
    return;
  }
  getGpuLockState().gGpuCv.wait(
      lock, [] { return !getGpuLockState().gGpuOwner.has_value(); });
  getGpuLockState().gGpuOwner = tid;
}

void unlockGpu() {
  if (gSemaphoreMode.load(std::memory_order_acquire)) {
    return;
  }
  std::thread::id tid = std::this_thread::get_id();
  std::unique_lock<std::mutex> lock(getGpuLockState().gGpuMutex);
  if (!getGpuLockState().gGpuOwner.has_value() ||
      getGpuLockState().gGpuOwner != tid) {
    return;
  }
  getGpuLockState().gGpuOwner = std::nullopt;
  lock.unlock();
  getGpuLockState().gGpuCv.notify_one();
}

} // namespace gluten
