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
package org.apache.gluten.gpu

import java.util.concurrent.TimeUnit

/**
 * Estimates GPU memory for a single running task using time-windowed blending.
 *
 * Initially returns the default estimate. As the task runs and actual GPU memory usage is measured,
 * the estimate gradually shifts from the default to the measured value over a configurable time
 * window (default: 100ms of active compute time).
 *
 * This avoids two problems:
 *   - Over-allocating permits for tasks that end up using less memory than expected
 *   - Under-allocating permits before we have enough measurement data
 */
private[gpu] class GpuTaskMemoryEstimator(
    val taskId: Long,
    val defaultEstimate: Long,
    val allowDynamicUpdate: Boolean) {

  private val startTimeNanos: Long = System.nanoTime()
  private var totalTimeLost: Long = 0
  private var maxMemory: Long = 0

  /**
   * Update with the latest measurement data.
   *
   * @param timeLost
   *   cumulative time the task spent blocked (not actively computing)
   * @param memory
   *   peak GPU memory usage observed so far
   */
  def update(timeLost: Long, memory: Long): Unit = {
    totalTimeLost = timeLost
    maxMemory = math.max(memory, maxMemory)
  }

  /** Get the current memory estimate for this task. */
  def estimate(): Long = {
    if (!allowDynamicUpdate) {
      return defaultEstimate
    }
    if (maxMemory > defaultEstimate) {
      maxMemory
    } else {
      val currentTime = System.nanoTime()
      val activeTime = currentTime - startTimeNanos - totalTimeLost
      val activePctOfWindow =
        math.max(math.min(activeTime / GpuTaskMemoryEstimator.TIME_WINDOW, 1.0), 0.0)
      val pctForDefault = 1.0 - activePctOfWindow
      (defaultEstimate * pctForDefault).toLong + (activePctOfWindow * maxMemory).toLong
    }
  }
}

private[gpu] object GpuTaskMemoryEstimator {
  val TIME_WINDOW: Double = TimeUnit.MILLISECONDS.toNanos(100).toDouble
}
