#pragma once
#include <cstdint>
#include <memory>

class GpuMemoryTracker {
public:
  static void initialize() {}
  static void shutdown() {}
  static void setCurrentTask(int64_t) {}
  static void clearCurrentTask() {}
  static GpuMemoryTracker* instance() { return nullptr; }
  int64_t getMaxTaskMemory(int64_t) { return 0; }
  int64_t clearTaskMemory(int64_t) { return 0; }
  static void setGpuSemaphoreMode(bool) {}
};
