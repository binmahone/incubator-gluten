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

/**
 * JNI bridge to the native GpuMemoryTracker. Wraps the RMM device memory resource to track per-task
 * GPU memory allocations.
 *
 * These native methods are defined in VeloxJniWrapper.cc under the GLUTEN_ENABLE_GPU guard.
 */
object GpuMemoryTrackerJniWrapper {

  /** Initialize the native memory tracker (wraps the current RMM resource). */
  @native def initialize(): Unit

  /** Shutdown and restore original RMM resource. */
  @native def shutdown(): Unit

  /** Associate the calling thread with the given task ID (for allocation tracking). */
  @native def setCurrentTask(taskId: Long): Unit

  /** Clear the task association for the calling thread. */
  @native def clearCurrentTask(): Unit

  /** Get peak GPU memory usage for the given task (bytes). Returns 0 if unknown. */
  @native def getMaxTaskMemory(taskId: Long): Long

  /** Clear tracking data for a completed task. Returns peak memory before removal. */
  @native def clearTaskMemory(taskId: Long): Long
}
