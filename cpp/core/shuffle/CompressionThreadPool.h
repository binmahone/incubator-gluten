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

// Persistent thread pool for shuffle compression. Eliminates per-call thread
// creation overhead (~100-200μs) by reusing pre-spawned worker threads.
// Thread-safe: multiple callers can invoke submitAndWait concurrently; each
// call blocks only until its own batch of tasks completes.
class CompressionThreadPool {
 public:
  explicit CompressionThreadPool(int numThreads) : stop_(false) {
    workers_.reserve(numThreads);
    for (int i = 0; i < numThreads; i++) {
      workers_.emplace_back([this] { workerLoop(); });
    }
  }

  ~CompressionThreadPool() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stop_ = true;
    }
    cv_.notify_all();
    for (auto& w : workers_) {
      w.join();
    }
  }

  CompressionThreadPool(const CompressionThreadPool&) = delete;
  CompressionThreadPool& operator=(const CompressionThreadPool&) = delete;

  // Submit a batch of tasks and block until all complete.
  void submitAndWait(std::vector<std::function<void()>>& tasks) {
    if (tasks.empty()) {
      return;
    }

    std::atomic<int32_t> remaining(static_cast<int32_t>(tasks.size()));
    std::mutex doneMutex;
    std::condition_variable doneCv;

    {
      std::lock_guard<std::mutex> lock(mutex_);
      for (auto& task : tasks) {
        queue_.emplace(Task{std::move(task), &remaining, &doneMutex, &doneCv});
      }
    }
    cv_.notify_all();

    std::unique_lock<std::mutex> doneLock(doneMutex);
    doneCv.wait(doneLock, [&] { return remaining.load(std::memory_order_acquire) == 0; });
  }

  int numThreads() const {
    return static_cast<int>(workers_.size());
  }

 private:
  struct Task {
    std::function<void()> fn;
    std::atomic<int32_t>* remaining;
    std::mutex* doneMutex;
    std::condition_variable* doneCv;
  };

  void workerLoop() {
    while (true) {
      Task task;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] { return stop_ || !queue_.empty(); });
        if (stop_ && queue_.empty()) {
          return;
        }
        task = std::move(queue_.front());
        queue_.pop();
      }

      task.fn();

      if (task.remaining->fetch_sub(1, std::memory_order_acq_rel) == 1) {
        std::lock_guard<std::mutex> lock(*task.doneMutex);
        task.doneCv->notify_one();
      }
    }
  }

  std::vector<std::thread> workers_;
  std::queue<Task> queue_;
  std::mutex mutex_;
  std::condition_variable cv_;
  bool stop_;
};

} // namespace gluten
