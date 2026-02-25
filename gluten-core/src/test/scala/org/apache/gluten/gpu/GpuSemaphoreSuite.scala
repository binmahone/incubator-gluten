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

import org.scalatest.BeforeAndAfterEach
import org.scalatest.funsuite.AnyFunSuite

class GpuSemaphoreSuite extends AnyFunSuite with BeforeAndAfterEach {

  override def afterEach(): Unit = {
    GpuSemaphore.reset()
    super.afterEach()
  }

  test("initialize with static config") {
    val gpuMemory = 1024L * 1024 * 1024 // 1 GB
    GpuSemaphore.initialize(gpuMemory, 4)
    assert(GpuSemaphore.isInitialized)
    assert(GpuSemaphore.activeTaskCount == 0)
  }

  test("initialize with full config") {
    val gpuMemory = 1024L * 1024 * 1024
    val memPerTask = gpuMemory / 4
    GpuSemaphore.initialize(gpuMemory, memPerTask, maxConcurrentTasks = 0, dynamicEnabled = false)
    assert(GpuSemaphore.isInitialized)
  }

  test("initialize with custom memory provider") {
    val gpuMemory = 1024L * 1024 * 1024
    val memPerTask = gpuMemory / 4
    val provider = new MockGpuMemoryProvider(Map.empty)
    GpuSemaphore.initialize(gpuMemory, memPerTask, 0, dynamicEnabled = true, provider)
    assert(GpuSemaphore.isInitialized)
  }

  test("not initialized until explicit init") {
    assert(!GpuSemaphore.isInitialized)
    GpuSemaphore.acquireIfNecessary(null)
    assert(!GpuSemaphore.isInitialized)
  }

  test("shutdown clears state") {
    GpuSemaphore.initialize(1024L * 1024 * 1024, 2)
    assert(GpuSemaphore.isInitialized)
    GpuSemaphore.shutdown()
    assert(!GpuSemaphore.isInitialized)
  }

  test("memToPermits calculation") {
    assert(GpuSemaphore.memToPermits(0) == 1)
    assert(GpuSemaphore.memToPermits(32L * 1024 * 1024) == 1)
    assert(GpuSemaphore.memToPermits(64L * 1024 * 1024) == 2)
    assert(GpuSemaphore.memToPermits(1024L * 1024 * 1024) == 32)
    assert(GpuSemaphore.memToPermits(80L * 1024 * 1024 * 1024) == 2560)
  }

  test("GpuStageMemoryEstimator returns default when not dynamic") {
    val estimator = new GpuStageMemoryEstimator(0, 100L, dynamicEnabled = false)
    assert(estimator.estimate() == 100L)
    estimator.addTaskIfNeeded(1)
    estimator.taskDone(1)
    assert(estimator.estimate() == 100L)
  }

  test("GpuStageMemoryEstimator returns default when no data") {
    val estimator = new GpuStageMemoryEstimator(0, 100L, dynamicEnabled = true)
    assert(estimator.estimate() == 100L)
  }

  test("GpuStageMemoryEstimator records completed tasks with default provider") {
    val estimator = new GpuStageMemoryEstimator(0, 100L, dynamicEnabled = true)
    for (i <- 1 to 10) {
      estimator.addTaskIfNeeded(i)
      estimator.taskDone(i)
    }
    assert(estimator.estimate() == 100L)
  }

  test("GpuStageMemoryEstimator uses real memory from provider") {
    val memoryData = (1L to 10L).map(id => id -> (50L * 1024 * 1024)).toMap
    val provider = new MockGpuMemoryProvider(memoryData)
    val defaultEstimate = 200L * 1024 * 1024
    val estimator =
      new GpuStageMemoryEstimator(0, defaultEstimate, dynamicEnabled = true, provider)

    for (id <- 1L to 10L) {
      estimator.addTaskIfNeeded(id)
      estimator.taskDone(id)
    }

    val est = estimator.estimate()
    assert(
      est < defaultEstimate,
      s"Dynamic estimate $est should be less than default $defaultEstimate")
    assert(est > 0, "Dynamic estimate should be positive")
  }

  test("GpuStageMemoryEstimator updates active task estimates") {
    val memoryData = Map(1L -> (100L * 1024 * 1024), 2L -> (200L * 1024 * 1024))
    val provider = new MockGpuMemoryProvider(memoryData)
    val estimator =
      new GpuStageMemoryEstimator(0, 256L * 1024 * 1024, dynamicEnabled = true, provider)

    estimator.addTaskIfNeeded(1L)
    estimator.addTaskIfNeeded(2L)

    val est = estimator.estimate()
    assert(est > 0, "Estimate with active tasks and provider data should be positive")
  }
}

class MockGpuMemoryProvider(memoryMap: Map[Long, Long]) extends GpuMemoryProvider {
  override def getMaxTaskGpuMemory(taskId: Long): Long = memoryMap.getOrElse(taskId, 0L)
  override def getTotalBlockedTime(taskId: Long): Long = 0L
}
