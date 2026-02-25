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

import org.apache.spark.TaskContext
import org.apache.spark.internal.Logging

import java.util.concurrent.ConcurrentHashMap

import scala.collection.mutable.ArrayBuffer

/**
 * Manages GPU task concurrency using a permit-based semaphore. Each permit represents a fixed
 * amount of GPU memory (32 MiB by default).
 *
 * Tasks acquire permits before using the GPU and release them upon completion. The number of
 * permits per task is determined by memory estimation, either static (from config) or dynamic
 * (based on actual measured usage per stage).
 */
object GpuSemaphore extends Logging {

  @volatile private var instance: GpuSemaphoreImpl = _

  private[gpu] val PERMIT_MEMORY_SIZE: Long = 32L * 1024 * 1024 // 32 MiB per permit

  def memToPermits(memory: Long): Long = math.max(1, memory / PERMIT_MEMORY_SIZE)

  /**
   * Initialize the GPU semaphore. Must be called once per executor before any task acquires.
   *
   * @param gpuMemorySize
   *   total GPU memory in bytes
   * @param defaultMemoryPerTask
   *   default estimated GPU memory per task in bytes (used for initial permit calculation)
   * @param maxConcurrentTasks
   *   hard limit on concurrent GPU tasks (0 = no limit, use permits only)
   * @param dynamicEnabled
   *   whether to enable dynamic memory estimation
   * @param memoryProvider
   *   provider for querying actual GPU memory usage per task
   */
  def initialize(
      gpuMemorySize: Long,
      defaultMemoryPerTask: Long,
      maxConcurrentTasks: Int,
      dynamicEnabled: Boolean,
      memoryProvider: GpuMemoryProvider = NoOpGpuMemoryProvider): Unit = synchronized {
    if (instance != null) {
      logWarning("GpuSemaphore already initialized, reinitializing.")
      instance.shutdown()
    }
    instance = new GpuSemaphoreImpl(
      gpuMemorySize,
      defaultMemoryPerTask,
      maxConcurrentTasks,
      dynamicEnabled,
      memoryProvider)
    logInfo(
      s"GpuSemaphore initialized: gpuMemory=${gpuMemorySize / (1024 * 1024)}MB, " +
        s"defaultPerTask=${defaultMemoryPerTask / (1024 * 1024)}MB, " +
        s"maxPermits=${memToPermits(gpuMemorySize)}, " +
        s"maxConcurrentTasks=$maxConcurrentTasks, dynamic=$dynamicEnabled, " +
        s"memoryProvider=${memoryProvider.getClass.getSimpleName}")
  }

  /**
   * Initialize with a simple static configuration (mainly for testing).
   *
   * @param gpuMemorySize
   *   total GPU memory in bytes
   * @param concurrentTasks
   *   desired number of concurrent tasks
   */
  def initialize(gpuMemorySize: Long, concurrentTasks: Int): Unit = {
    val memPerTask = if (concurrentTasks > 0) gpuMemorySize / concurrentTasks else gpuMemorySize
    initialize(gpuMemorySize, memPerTask, concurrentTasks, dynamicEnabled = false)
  }

  private def getInstance: GpuSemaphoreImpl = {
    if (instance == null) {
      GpuSemaphore.synchronized {
        if (instance == null) {
          logWarning("GpuSemaphore used before initialization, defaulting to 1 concurrent task.")
          initialize(PERMIT_MEMORY_SIZE, 1)
        }
      }
    }
    instance
  }

  /**
   * Acquire the GPU semaphore for the given task context. If the task already holds the semaphore,
   * this is a no-op (reentrant).
   */
  def acquireIfNecessary(context: TaskContext): Unit = {
    if (context != null) {
      getInstance.acquireIfNecessary(context)
    }
  }

  /**
   * Release the GPU semaphore for the given task context. Safe to call even if the task doesn't
   * hold the semaphore.
   */
  def releaseIfNecessary(context: TaskContext): Unit = {
    if (context != null) {
      getInstance.releaseIfNecessary(context)
    }
  }

  /** Check whether the semaphore has been initialized. */
  def isInitialized: Boolean = instance != null

  /** Shutdown the GPU semaphore. Does not wait for active tasks. */
  def shutdown(): Unit = synchronized {
    if (instance != null) {
      instance.shutdown()
      instance = null
    }
  }

  // Visible for testing
  private[gpu] def activeTaskCount: Long =
    if (instance != null) instance.activeTaskCount else 0

  private[gpu] def waitingCount: Int =
    if (instance != null) instance.waitingCount else 0

  private[gpu] def reset(): Unit = synchronized {
    if (instance != null) {
      instance.shutdown()
      instance = null
    }
  }
}

final private class GpuSemaphoreImpl(
    gpuMemorySize: Long,
    defaultMemoryPerTask: Long,
    maxConcurrentTasks: Int,
    dynamicEnabled: Boolean,
    memoryProvider: GpuMemoryProvider)
  extends Logging {

  private val maxPermits = GpuSemaphore.memToPermits(gpuMemorySize)
  private val semaphore = new PrioritySemaphore(maxPermits, maxConcurrentTasks)

  private val tasks = new ConcurrentHashMap[Long, TaskSemaphoreInfo]

  private val stageEstimators = {
    val lru = new java.util.LinkedHashMap[Int, GpuStageMemoryEstimator]() {
      override def removeEldestEntry(
          entry: java.util.Map.Entry[Int, GpuStageMemoryEstimator]): Boolean = {
        size > 100
      }
    }
    java.util.Collections.synchronizedMap(lru)
  }

  def acquireIfNecessary(context: TaskContext): Unit = {
    val taskAttemptId = context.taskAttemptId()
    val stageId = context.stageId()

    val stageEstimate = stageEstimators.computeIfAbsent(
      stageId,
      _ =>
        new GpuStageMemoryEstimator(stageId, defaultMemoryPerTask, dynamicEnabled, memoryProvider))

    val taskInfo = tasks.computeIfAbsent(
      taskAttemptId,
      _ => {
        val info = new TaskSemaphoreInfo(stageId, taskAttemptId, stageEstimate)
        logDebug(s"Registering task $taskAttemptId (stage $stageId) with GPU semaphore")
        info
      }
    )
    taskInfo.blockUntilReady(semaphore)
    stageEstimate.addTaskIfNeeded(taskAttemptId)
  }

  def releaseIfNecessary(context: TaskContext): Unit = {
    val taskAttemptId = context.taskAttemptId()
    val taskInfo = tasks.remove(taskAttemptId)
    if (taskInfo != null) {
      taskInfo.releaseSemaphore(semaphore)
      val estimator = stageEstimators.get(taskInfo.stageId)
      if (estimator != null) {
        estimator.taskDone(taskAttemptId)
      }
      logDebug(s"Released GPU semaphore for task $taskAttemptId (stage ${taskInfo.stageId})")
    }
  }

  def activeTaskCount: Long = semaphore.activeTaskCount
  def waitingCount: Int = semaphore.waitingCount

  def shutdown(): Unit = {
    if (!tasks.isEmpty) {
      logDebug(s"Shutting down GpuSemaphore with ${tasks.size} tasks still registered")
    }
    tasks.clear()
  }
}

/**
 * Tracks semaphore state for a single task. A task either holds the semaphore or is waiting for it.
 */
final private class TaskSemaphoreInfo(
    val stageId: Int,
    val taskAttemptId: Long,
    stageEstimator: GpuStageMemoryEstimator)
  extends Logging {

  @volatile private var hasSemaphore = false
  private var permitsUsed: Long = 0

  def blockUntilReady(semaphore: PrioritySemaphore): Unit = synchronized {
    if (hasSemaphore) return

    val used = semaphore.acquire(
      () => GpuSemaphore.memToPermits(stageEstimator.estimate()),
      priority = taskAttemptId,
      taskId = taskAttemptId
    )
    permitsUsed = used
    hasSemaphore = true
  }

  def releaseSemaphore(semaphore: PrioritySemaphore): Unit = synchronized {
    if (hasSemaphore) {
      semaphore.release(permitsUsed)
      hasSemaphore = false
    }
  }
}

/**
 * Estimates GPU memory usage per task at the stage level. Starts with a default estimate and
 * adjusts based on actual measured memory usage as tasks complete.
 *
 * Uses the 80th percentile of completed task memory (combined with active task estimates) to handle
 * variance across tasks. When a GpuMemoryProvider is available, actual peak GPU memory from
 * completed tasks is recorded; otherwise falls back to the default estimate.
 */
private[gpu] class GpuStageMemoryEstimator(
    val stageId: Int,
    private val defaultEstimate: Long,
    val dynamicEnabled: Boolean,
    private val memoryProvider: GpuMemoryProvider = NoOpGpuMemoryProvider)
  extends Logging {

  private val completedStats = new StatEstimator(4, defaultEstimate.toDouble)
  private val activeTasks =
    new java.util.HashMap[Long, GpuTaskMemoryEstimator]()

  def addTaskIfNeeded(taskId: Long): Unit = synchronized {
    activeTasks.computeIfAbsent(
      taskId,
      _ => {
        val currentDefault = estimate()
        new GpuTaskMemoryEstimator(taskId, currentDefault, dynamicEnabled)
      })
  }

  def taskDone(taskId: Long): Unit = synchronized {
    val taskEstimator = activeTasks.remove(taskId)
    if (taskEstimator != null) {
      val maxMemory = memoryProvider.getMaxTaskGpuMemory(taskId)
      if (maxMemory > 0) {
        completedStats.add(maxMemory.toDouble)
      } else {
        completedStats.add(taskEstimator.defaultEstimate.toDouble)
      }
    }
  }

  def estimate(): Long = synchronized {
    if (!dynamicEnabled) {
      return defaultEstimate
    }

    updateActiveEstimates()
    val activeValues = ArrayBuffer.empty[Double]
    activeTasks.values().forEach(est => activeValues += est.estimate().toDouble)
    completedStats.percentile(0.8, activeValues).toLong
  }

  private def updateActiveEstimates(): Unit = {
    activeTasks.forEach {
      (taskId, estimator) =>
        val maxMemory = memoryProvider.getMaxTaskGpuMemory(taskId)
        val timeLost = memoryProvider.getTotalBlockedTime(taskId)
        estimator.update(timeLost, maxMemory)
    }
  }
}

/**
 * Computes percentile statistics over a buffer of observed values. Pads with a default value when
 * fewer than `minEntries` samples are available.
 */
private[gpu] class StatEstimator(minEntries: Int, defaultValue: Double) {
  require(minEntries > 0, "Minimum entries must be positive")

  private val values = ArrayBuffer.empty[Double]

  def add(value: Double): Unit = {
    values += value
    if (values.length > 200) {
      values.remove(0, values.length - 200)
    }
  }

  def percentile(p: Double, others: ArrayBuffer[Double]): Double = {
    require(p >= 0 && p <= 1, "Percentile must be between 0-1")

    if (values.isEmpty && others.isEmpty) return defaultValue

    val combined = values ++ others
    while (combined.length < minEntries) {
      combined += defaultValue
    }
    val padded = combined.sorted

    val n = padded.size
    val pos = p * (n + 1)

    pos match {
      case _ if pos < 1 => padded.head
      case _ if pos >= n => padded.last
      case _ =>
        val k = math.floor(pos).toInt
        val lower = padded(k - 1)
        val upper = padded(k)
        val fraction = pos - k
        lower + fraction * (upper - lower)
    }
  }

  // Visible for testing
  private[gpu] def size: Int = values.size
}
