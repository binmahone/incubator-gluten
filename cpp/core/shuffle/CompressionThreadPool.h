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

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace gluten {

// Persistent thread pool for shuffle compression. Reuses pre-spawned workers
// to avoid per-batch thread creation overhead (critical when 100+ Spark tasks
// run concurrently). Thread-safe for concurrent callers (e.g. main shuffle
// path + memory reclamation): submitMutex_ serializes submissions so only
// one batch is in-flight at a time. doneMutex_/doneCv_ are pool members
// (not stack-local) to avoid the lifetime/destruction race that caused
// pthread assertion failures.
class CompressionThreadPool {
 public:
  explicit CompressionThreadPool(int numThreads) : stop_(false), remaining_(0) {
    workers_.reserve(numThreads);
    for (int i = 0; i < numThreads; i++) {
      workers_.emplace_back([this] { workerLoop(); });
    }
  }

  ~CompressionThreadPool() {
    {
      std::lock_guard<std::mutex> lock(queueMutex_);
      stop_ = true;
    }
    queueCv_.notify_all();
    for (auto& w : workers_) {
      w.join();
    }
  }

  CompressionThreadPool(const CompressionThreadPool&) = delete;
  CompressionThreadPool& operator=(const CompressionThreadPool&) = delete;

  void submitAndWait(std::vector<std::function<void()>>& tasks) {
    if (tasks.empty()) {
      return;
    }

    // Serialize concurrent callers (main thread + memory reclamation).
    std::lock_guard<std::mutex> submitLock(submitMutex_);

    remaining_.store(static_cast<int32_t>(tasks.size()), std::memory_order_release);

    {
      std::lock_guard<std::mutex> lock(queueMutex_);
      for (auto& task : tasks) {
        queue_.emplace(std::move(task));
      }
    }
    queueCv_.notify_all();

    // Wait for all tasks to complete. doneMutex_/doneCv_ are pool members
    // (stable lifetime), safe because submitMutex_ ensures only one batch.
    std::unique_lock<std::mutex> doneLock(doneMutex_);
    doneCv_.wait(doneLock, [this] { return remaining_.load(std::memory_order_acquire) == 0; });
  }

  int numThreads() const {
    return static_cast<int>(workers_.size());
  }

 private:
  void workerLoop() {
    while (true) {
      std::function<void()> fn;
      {
        std::unique_lock<std::mutex> lock(queueMutex_);
        queueCv_.wait(lock, [this] { return stop_ || !queue_.empty(); });
        if (stop_ && queue_.empty()) {
          return;
        }
        fn = std::move(queue_.front());
        queue_.pop();
      }

      fn();

      if (remaining_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        std::lock_guard<std::mutex> lock(doneMutex_);
        doneCv_.notify_one();
      }
    }
  }

  std::mutex submitMutex_;
  bool stop_;
  std::atomic<int32_t> remaining_;
  std::vector<std::thread> workers_;
  std::queue<std::function<void()>> queue_;
  std::mutex queueMutex_;
  std::condition_variable queueCv_;
  std::mutex doneMutex_;
  std::condition_variable doneCv_;
};

} // namespace gluten
