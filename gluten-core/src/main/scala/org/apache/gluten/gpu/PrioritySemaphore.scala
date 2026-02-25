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

import java.util.PriorityQueue
import java.util.concurrent.locks.{Condition, ReentrantLock}

/**
 * A permit-based semaphore with priority ordering for GPU task scheduling.
 *
 * Waiting threads are ordered by priority (lower value = higher priority), with task ID as
 * tiebreaker. Supports both permit-based limits (memory) and a hard concurrent task count limit.
 *
 * @param maxPermits
 *   total available permits (typically mapped from GPU memory, e.g. 1 permit = 32 MiB)
 * @param maxConcurrentTasks
 *   hard limit on concurrent tasks (0 = no limit)
 */
class PrioritySemaphore(val maxPermits: Long, val maxConcurrentTasks: Int) {

  private val lock = new ReentrantLock()
  private var occupiedPermits: Long = 0
  private var activeTasks: Long = 0

  private case class WaiterInfo(
      priority: Long,
      taskId: Long,
      condition: Condition,
      computePermits: () => Long) {
    var signaled: Boolean = false
    var permitsUsed: Long = 0
  }

  private val waiterComparator: java.util.Comparator[WaiterInfo] =
    (a: WaiterInfo, b: WaiterInfo) => {
      val cmp = java.lang.Long.compare(a.priority, b.priority)
      if (cmp != 0) cmp else java.lang.Long.compare(a.taskId, b.taskId)
    }

  private val waitingQueue = new PriorityQueue[WaiterInfo](waiterComparator)

  def tryAcquire(numPermits: Long, priority: Long, taskId: Long): Boolean = {
    lock.lock()
    try {
      if (waitingQueue.size() > 0) {
        val head = waitingQueue.peek()
        val cmp = waiterComparator.compare(
          WaiterInfo(priority, taskId, null, null),
          head
        )
        if (cmp > 0) return false
      }
      if (!canAcquire(numPermits)) return false
      commitAcquire(numPermits)
      true
    } finally {
      lock.unlock()
    }
  }

  /**
   * Acquire permits, blocking until available. The `computePermits` function is re-evaluated when
   * this thread is woken up, allowing dynamic permit adjustment.
   *
   * @return
   *   the number of permits actually acquired
   */
  def acquire(computePermits: () => Long, priority: Long, taskId: Long): Long = {
    lock.lock()
    try {
      val numPermitsNow = computePermits()
      if (tryAcquire(numPermitsNow, priority, taskId)) {
        numPermitsNow
      } else {
        val condition = lock.newCondition()
        val info = WaiterInfo(priority, taskId, condition, computePermits)
        try {
          waitingQueue.add(info)
          while (!info.signaled) {
            info.condition.await()
          }
          info.permitsUsed
        } catch {
          case e: Exception =>
            waitingQueue.remove(info)
            if (info.signaled) {
              release(info.permitsUsed)
            }
            throw e
        }
      }
    } finally {
      lock.unlock()
    }
  }

  def release(numPermits: Long): Unit = {
    lock.lock()
    try {
      occupiedPermits -= numPermits
      activeTasks -= 1
      var done = false
      while (!done && waitingQueue.size() > 0) {
        val next = waitingQueue.peek()
        val threadPermits = next.computePermits()
        if (canAcquire(threadPermits)) {
          val popped = waitingQueue.poll()
          assert(popped eq next)
          commitAcquire(threadPermits)
          next.signaled = true
          next.permitsUsed = threadPermits
          next.condition.signal()
        } else {
          done = true
        }
      }
    } finally {
      lock.unlock()
    }
  }

  def availablePermits: Long = {
    lock.lock()
    try { maxPermits - occupiedPermits }
    finally { lock.unlock() }
  }

  def activeTaskCount: Long = {
    lock.lock()
    try { activeTasks }
    finally { lock.unlock() }
  }

  def waitingCount: Int = {
    lock.lock()
    try { waitingQueue.size() }
    finally { lock.unlock() }
  }

  private def commitAcquire(numPermits: Long): Unit = {
    occupiedPermits += numPermits
    activeTasks += 1
  }

  private def canAcquire(numPermits: Long): Boolean = {
    val hasPermits = occupiedPermits + numPermits <= maxPermits
    val withinTaskLimit =
      maxConcurrentTasks <= 0 || activeTasks < maxConcurrentTasks
    hasPermits && withinTaskLimit
  }
}
