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

import org.apache.spark.internal.Logging

/**
 * GpuMemoryProvider backed by the native GpuMemoryTracker (RMM-based per-task tracking via JNI).
 *
 * When a task completes, the peak memory is retrieved and the native tracking data is cleared via
 * `clearTaskMemory`. This avoids unbounded growth of the native tracking map.
 */
class RmmGpuMemoryProvider extends GpuMemoryProvider with Logging {

  override def getMaxTaskGpuMemory(taskId: Long): Long = {
    try {
      GpuMemoryTrackerJniWrapper.getMaxTaskMemory(taskId)
    } catch {
      case e: UnsatisfiedLinkError =>
        logWarning(s"GpuMemoryTracker native method not available: ${e.getMessage}")
        0L
    }
  }

  override def getTotalBlockedTime(taskId: Long): Long = 0L
}
