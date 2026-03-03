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
package org.apache.gluten.gpu;

/**
 * JNI bridge to the native GpuMemoryTracker. Wraps the RMM device memory resource to track per-task
 * GPU memory allocations.
 *
 * <p>Must be a Java class (not Scala object) so that {@code static native} methods produce JNI
 * symbols without the {@code _00024} ($) encoding that Scala objects require.
 *
 * <p>Native methods are defined in VeloxJniWrapper.cc under the GLUTEN_ENABLE_GPU guard.
 */
public final class GpuMemoryTrackerJniWrapper {

  private GpuMemoryTrackerJniWrapper() {}

  /** Initialize the native memory tracker (wraps the current RMM resource). */
  public static native void initialize();

  /** Shutdown and restore original RMM resource. */
  public static native void shutdown();

  /** Associate the calling thread with the given task ID (for allocation tracking). */
  public static native void setCurrentTask(long taskId);

  /** Clear the task association for the calling thread. */
  public static native void clearCurrentTask();

  /** Get peak GPU memory usage for the given task (bytes). Returns 0 if unknown. */
  public static native long getMaxTaskMemory(long taskId);

  /** Clear tracking data for a completed task. Returns peak memory before removal. */
  public static native long clearTaskMemory(long taskId);

  /** Set the max number of concurrent GPU operations in C++ GpuLock (counting semaphore). */
  public static native void setMaxConcurrentGpuTasks(int maxTasks);

  /** Query total device memory in bytes via cudaMemGetInfo + cudaGetDeviceProperties. Returns 0 on failure. */
  public static native long getDeviceMemorySize();
}
