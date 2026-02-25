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

import scala.collection.mutable.ArrayBuffer

class GpuTaskMemoryEstimatorSuite extends AnyFunSuite {

  private val MB = 1024L * 1024

  test("returns default estimate when dynamic is disabled") {
    val est = new GpuTaskMemoryEstimator(1, 256 * MB, allowDynamicUpdate = false)
    assert(est.estimate() == 256 * MB)
    est.update(0, 100 * MB)
    assert(est.estimate() == 256 * MB)
  }

  test("returns default initially when dynamic is enabled but no measurement") {
    val est = new GpuTaskMemoryEstimator(1, 256 * MB, allowDynamicUpdate = true)
    val initialEst = est.estimate()
    assert(initialEst >= 200 * MB, s"Initial estimate $initialEst should be close to default")
  }

  test("returns measured memory when it exceeds default") {
    val est = new GpuTaskMemoryEstimator(1, 256 * MB, allowDynamicUpdate = true)
    est.update(0, 512 * MB)
    assert(est.estimate() == 512 * MB)
  }

  test("blends toward measured value as active time increases") {
    val est = new GpuTaskMemoryEstimator(1, 256 * MB, allowDynamicUpdate = true)
    est.update(0, 100 * MB)

    Thread.sleep(50)
    val midEst = est.estimate()

    Thread.sleep(100)
    val lateEst = est.estimate()

    assert(lateEst <= midEst, s"Later estimate $lateEst should be <= mid estimate $midEst")
    assert(lateEst <= 256 * MB, "Estimate should not exceed default when measured is lower")
  }

  test("time lost reduces effective active time") {
    val est = new GpuTaskMemoryEstimator(1, 256 * MB, allowDynamicUpdate = true)
    est.update(0, 100 * MB)

    Thread.sleep(150)
    val estNoLoss = est.estimate()

    val est2 = new GpuTaskMemoryEstimator(2, 256 * MB, allowDynamicUpdate = true)
    est2.update(0, 100 * MB)

    Thread.sleep(150)
    val nanos150ms = 150L * 1000 * 1000
    est2.update(nanos150ms, 100 * MB)
    val estWithLoss = est2.estimate()

    assert(
      estWithLoss >= estNoLoss,
      s"Estimate with time loss ($estWithLoss) should be >= estimate without ($estNoLoss)")
  }

  test("max memory is tracked across updates") {
    val est = new GpuTaskMemoryEstimator(1, 256 * MB, allowDynamicUpdate = true)
    est.update(0, 100 * MB)
    est.update(0, 300 * MB)
    est.update(0, 200 * MB)

    assert(est.estimate() == 300 * MB, "Should keep the max observed memory")
  }
}

class StatEstimatorSuite extends AnyFunSuite {

  test("returns default when no values") {
    val stat = new StatEstimator(4, 100.0)
    assert(stat.percentile(0.8, ArrayBuffer.empty) == 100.0)
  }

  test("pads with default when fewer than minEntries") {
    val stat = new StatEstimator(4, 100.0)
    stat.add(50.0)
    val result = stat.percentile(0.8, ArrayBuffer.empty)
    assert(result >= 50.0 && result <= 100.0, s"Percentile $result should be between 50 and 100")
  }

  test("percentile with enough samples") {
    val stat = new StatEstimator(2, 0.0)
    for (v <- 1.0 to 10.0 by 1.0) {
      stat.add(v)
    }
    val p80 = stat.percentile(0.8, ArrayBuffer.empty)
    assert(p80 >= 8.0 && p80 <= 9.0, s"80th percentile $p80 should be around 8-9")
  }

  test("percentile 0 returns minimum") {
    val stat = new StatEstimator(1, 0.0)
    stat.add(10.0)
    stat.add(20.0)
    stat.add(30.0)
    val p0 = stat.percentile(0.0, ArrayBuffer.empty)
    assert(p0 == 10.0, s"0th percentile $p0 should be the minimum")
  }

  test("percentile 1 returns maximum") {
    val stat = new StatEstimator(1, 0.0)
    stat.add(10.0)
    stat.add(20.0)
    stat.add(30.0)
    val p100 = stat.percentile(1.0, ArrayBuffer.empty)
    assert(p100 == 30.0, s"100th percentile $p100 should be the maximum")
  }

  test("combines with others array") {
    val stat = new StatEstimator(2, 0.0)
    stat.add(10.0)
    stat.add(20.0)
    val others = ArrayBuffer(30.0, 40.0)
    val p80 = stat.percentile(0.8, others)
    assert(p80 >= 30.0 && p80 <= 40.0, s"P80 of [10,20,30,40] = $p80, expected around 34")
  }

  test("evicts old entries beyond 200 limit") {
    val stat = new StatEstimator(1, 0.0)
    for (i <- 1 to 250) {
      stat.add(i.toDouble)
    }
    assert(stat.size == 200, s"Size ${stat.size} should be capped at 200")
    val p80 = stat.percentile(0.8, ArrayBuffer.empty)
    assert(p80 > 50, "After eviction, early values should be gone")
  }
}
