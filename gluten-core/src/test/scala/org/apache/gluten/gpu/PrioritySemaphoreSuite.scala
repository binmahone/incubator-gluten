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

import org.scalatest.funsuite.AnyFunSuite

import java.util.concurrent.{CountDownLatch, Executors, TimeUnit}
import java.util.concurrent.atomic.AtomicInteger

class PrioritySemaphoreSuite extends AnyFunSuite {

  test("basic acquire and release") {
    val sem = new PrioritySemaphore(100, 0)
    assert(sem.availablePermits == 100)
    assert(sem.activeTaskCount == 0)

    val used = sem.acquire(() => 30, priority = 0, taskId = 1)
    assert(used == 30)
    assert(sem.availablePermits == 70)
    assert(sem.activeTaskCount == 1)

    sem.release(30)
    assert(sem.availablePermits == 100)
    assert(sem.activeTaskCount == 0)
  }

  test("tryAcquire succeeds when enough permits") {
    val sem = new PrioritySemaphore(100, 0)
    assert(sem.tryAcquire(50, priority = 0, taskId = 1))
    assert(sem.availablePermits == 50)
    assert(sem.tryAcquire(50, priority = 0, taskId = 2))
    assert(sem.availablePermits == 0)

    sem.release(50)
    sem.release(50)
  }

  test("tryAcquire fails when not enough permits") {
    val sem = new PrioritySemaphore(100, 0)
    assert(sem.tryAcquire(80, priority = 0, taskId = 1))
    assert(!sem.tryAcquire(30, priority = 0, taskId = 2))
    sem.release(80)
  }

  test("maxConcurrentTasks limit is enforced") {
    val sem = new PrioritySemaphore(1000, 2)
    assert(sem.tryAcquire(1, priority = 0, taskId = 1))
    assert(sem.tryAcquire(1, priority = 0, taskId = 2))
    assert(!sem.tryAcquire(1, priority = 0, taskId = 3))

    sem.release(1)
    assert(sem.tryAcquire(1, priority = 0, taskId = 3))

    sem.release(1)
    sem.release(1)
  }

  test("blocking acquire waits for permits") {
    val sem = new PrioritySemaphore(100, 0)
    sem.acquire(() => 80, priority = 0, taskId = 1)

    val acquired = new CountDownLatch(1)
    val thread = new Thread(
      () => {
        sem.acquire(() => 50, priority = 0, taskId = 2)
        acquired.countDown()
      })
    thread.start()

    Thread.sleep(100)
    assert(acquired.getCount == 1)
    assert(sem.waitingCount == 1)

    sem.release(80)
    assert(acquired.await(5, TimeUnit.SECONDS))
    assert(sem.activeTaskCount == 1)

    sem.release(50)
  }

  test("priority ordering: lower priority value acquires first") {
    val sem = new PrioritySemaphore(100, 1)
    sem.acquire(() => 10, priority = 0, taskId = 0)

    val order = new java.util.concurrent.ConcurrentLinkedQueue[Long]()
    val allStarted = new CountDownLatch(2)
    val allDone = new CountDownLatch(2)

    val lowPriority = new Thread(
      () => {
        allStarted.countDown()
        sem.acquire(() => 10, priority = 100, taskId = 2)
        order.add(2L)
        sem.release(10)
        allDone.countDown()
      })

    val highPriority = new Thread(
      () => {
        allStarted.countDown()
        sem.acquire(() => 10, priority = 1, taskId = 1)
        order.add(1L)
        sem.release(10)
        allDone.countDown()
      })

    lowPriority.start()
    Thread.sleep(50)
    highPriority.start()
    allStarted.await(5, TimeUnit.SECONDS)
    Thread.sleep(100)

    sem.release(10)
    assert(allDone.await(5, TimeUnit.SECONDS))

    val first = order.poll()
    val second = order.poll()
    assert(first == 1L, s"Expected high priority (1) first, got $first")
    assert(second == 2L, s"Expected low priority (2) second, got $second")
  }

  test("concurrent acquire and release stress test") {
    val sem = new PrioritySemaphore(100, 0)
    val numThreads = 20
    val iterations = 50
    val errors = new AtomicInteger(0)
    val executor = Executors.newFixedThreadPool(numThreads)
    val latch = new CountDownLatch(numThreads)

    for (i <- 0 until numThreads) {
      executor.submit(new Runnable {
        override def run(): Unit = {
          try {
            for (_ <- 0 until iterations) {
              val permits = (i % 10) + 1L
              sem.acquire(() => permits, priority = i.toLong, taskId = i.toLong)
              Thread.sleep(1)
              sem.release(permits)
            }
          } catch {
            case _: Exception => errors.incrementAndGet()
          } finally {
            latch.countDown()
          }
        }
      })
    }

    assert(latch.await(30, TimeUnit.SECONDS), "Stress test timed out")
    assert(errors.get() == 0, s"${errors.get()} errors during stress test")
    assert(sem.availablePermits == 100)
    assert(sem.activeTaskCount == 0)
    executor.shutdown()
  }

  test("dynamic permit recomputation on wakeup") {
    val sem = new PrioritySemaphore(100, 0)
    sem.acquire(() => 90, priority = 0, taskId = 0)

    var dynamicPermits = 50L
    val acquired = new CountDownLatch(1)
    val thread = new Thread(
      () => {
        sem.acquire(() => dynamicPermits, priority = 0, taskId = 1)
        acquired.countDown()
      })
    thread.start()
    Thread.sleep(100)

    dynamicPermits = 20L
    sem.release(90)

    assert(acquired.await(5, TimeUnit.SECONDS))
    assert(sem.availablePermits == 80)
    sem.release(20)
  }
}
