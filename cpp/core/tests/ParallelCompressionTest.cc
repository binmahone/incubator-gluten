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

#include "shuffle/CompressionThreadPool.h"
#include "shuffle/Payload.h"

#include <arrow/buffer.h>
#include <arrow/io/memory.h>
#include <arrow/memory_pool.h>
#include <arrow/util/compression.h>
#include <gtest/gtest.h>

#include <random>
#include <vector>

namespace gluten {

class ParallelCompressionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    pool_ = arrow::default_memory_pool();
    auto codecResult = arrow::util::Codec::Create(arrow::Compression::LZ4_FRAME);
    ASSERT_TRUE(codecResult.ok());
    codec_ = std::move(*codecResult);
    threadPool_ = std::make_unique<CompressionThreadPool>(4);
  }

  std::shared_ptr<arrow::Buffer> makeRandomBuffer(int64_t size, uint32_t seed = 42) {
    auto result = arrow::AllocateResizableBuffer(size, pool_);
    EXPECT_TRUE(result.ok());
    auto buf = std::move(*result);
    std::mt19937 gen(seed);
    std::uniform_int_distribution<uint8_t> dist(0, 255);
    auto* data = buf->mutable_data();
    for (int64_t i = 0; i < size; i++) {
      data[i] = dist(gen);
    }
    return std::move(buf);
  }

  std::shared_ptr<arrow::Buffer> makeCompressibleBuffer(int64_t size, uint32_t seed = 42) {
    auto result = arrow::AllocateResizableBuffer(size, pool_);
    EXPECT_TRUE(result.ok());
    auto buf = std::move(*result);
    std::mt19937 gen(seed);
    std::uniform_int_distribution<uint8_t> dist(0, 10);
    auto* data = buf->mutable_data();
    for (int64_t i = 0; i < size; i++) {
      data[i] = dist(gen);
    }
    return std::move(buf);
  }

  // Serialize a BlockPayload and return the raw bytes.
  std::shared_ptr<arrow::Buffer> serializePayload(std::unique_ptr<BlockPayload>& payload) {
    auto sink = arrow::io::BufferOutputStream::Create(1024 * 1024, pool_);
    EXPECT_TRUE(sink.ok());
    auto os = *sink;
    auto status = payload->serialize(os.get());
    EXPECT_TRUE(status.ok());
    auto result = os->Finish();
    EXPECT_TRUE(result.ok());
    return *result;
  }

  arrow::MemoryPool* pool_;
  std::unique_ptr<arrow::util::Codec> codec_;
  std::unique_ptr<CompressionThreadPool> threadPool_;
};

TEST_F(ParallelCompressionTest, SingleThreadBaseline) {
  std::vector<std::shared_ptr<arrow::Buffer>> buffers;
  std::vector<bool> isValidity = {false, false, false};
  for (int i = 0; i < 3; i++) {
    buffers.push_back(makeCompressibleBuffer(64 * 1024, i));
  }

  auto result = BlockPayload::fromBuffers(
      Payload::kCompressed, 100, std::move(buffers), &isValidity, pool_, codec_.get(), 1);
  ASSERT_TRUE(result.ok());
  auto payload = std::move(*result);
  ASSERT_NE(payload, nullptr);
  EXPECT_EQ(payload->type(), Payload::kCompressed);
  EXPECT_GT(payload->getCompressTime(), 0);
}

TEST_F(ParallelCompressionTest, ParallelProducesSameResult) {
  const int numBuffers = 8;
  const int64_t bufferSize = 128 * 1024;
  std::vector<bool> isValidity(numBuffers, false);

  // Create two identical copies of buffers.
  std::vector<std::shared_ptr<arrow::Buffer>> buffers1;
  std::vector<std::shared_ptr<arrow::Buffer>> buffers2;
  for (int i = 0; i < numBuffers; i++) {
    auto original = makeCompressibleBuffer(bufferSize, i);
    auto copy = arrow::AllocateResizableBuffer(bufferSize, pool_);
    ASSERT_TRUE(copy.ok());
    memcpy((*copy)->mutable_data(), original->data(), bufferSize);
    buffers1.push_back(original);
    buffers2.push_back(std::move(*copy));
  }

  // Compress with single thread.
  auto result1 = BlockPayload::fromBuffers(
      Payload::kCompressed, 200, std::move(buffers1), &isValidity, pool_, codec_.get(), 1);
  ASSERT_TRUE(result1.ok());
  auto payload1 = std::move(*result1);

  // Compress with 4 threads using persistent pool.
  auto result2 = BlockPayload::fromBuffers(
      Payload::kCompressed, 200, std::move(buffers2), &isValidity, pool_, codec_.get(), 4, threadPool_.get());
  ASSERT_TRUE(result2.ok());
  auto payload2 = std::move(*result2);

  // Serialize both and compare bytes.
  auto bytes1 = serializePayload(payload1);
  auto bytes2 = serializePayload(payload2);

  ASSERT_EQ(bytes1->size(), bytes2->size());
  EXPECT_EQ(memcmp(bytes1->data(), bytes2->data(), bytes1->size()), 0);
}

TEST_F(ParallelCompressionTest, ParallelWithNullAndEmptyBuffers) {
  std::vector<std::shared_ptr<arrow::Buffer>> buffers;
  std::vector<bool> isValidity = {false, false, false, false, false};

  buffers.push_back(makeCompressibleBuffer(64 * 1024, 0));
  buffers.push_back(nullptr);
  buffers.push_back(makeCompressibleBuffer(128 * 1024, 2));
  buffers.push_back(std::make_shared<arrow::Buffer>(nullptr, 0));
  buffers.push_back(makeCompressibleBuffer(32 * 1024, 4));

  auto result = BlockPayload::fromBuffers(
      Payload::kCompressed, 50, std::move(buffers), &isValidity, pool_, codec_.get(), 3, threadPool_.get());
  ASSERT_TRUE(result.ok());
  auto payload = std::move(*result);
  ASSERT_NE(payload, nullptr);
  EXPECT_EQ(payload->type(), Payload::kCompressed);
}

TEST_F(ParallelCompressionTest, ParallelWithSingleBuffer) {
  std::vector<std::shared_ptr<arrow::Buffer>> buffers;
  std::vector<bool> isValidity = {false};
  buffers.push_back(makeCompressibleBuffer(64 * 1024, 0));

  auto result = BlockPayload::fromBuffers(
      Payload::kCompressed, 10, std::move(buffers), &isValidity, pool_, codec_.get(), 4, threadPool_.get());
  ASSERT_TRUE(result.ok());
  auto payload = std::move(*result);
  ASSERT_NE(payload, nullptr);
}

TEST_F(ParallelCompressionTest, UncompressedBypassesParallel) {
  std::vector<std::shared_ptr<arrow::Buffer>> buffers;
  std::vector<bool> isValidity = {false, false};
  buffers.push_back(makeRandomBuffer(1024, 0));
  buffers.push_back(makeRandomBuffer(2048, 1));

  auto result = BlockPayload::fromBuffers(
      Payload::kUncompressed, 10, std::move(buffers), &isValidity, pool_, codec_.get(), 4, threadPool_.get());
  ASSERT_TRUE(result.ok());
  auto payload = std::move(*result);
  EXPECT_EQ(payload->type(), Payload::kUncompressed);
}

TEST_F(ParallelCompressionTest, ParallelDeserializationRoundTrip) {
  const int numBuffers = 6;
  const int64_t bufferSize = 64 * 1024;
  std::vector<bool> isValidity(numBuffers, false);

  // Save original data for comparison.
  std::vector<std::vector<uint8_t>> originalData(numBuffers);
  std::vector<std::shared_ptr<arrow::Buffer>> buffers;
  for (int i = 0; i < numBuffers; i++) {
    auto buf = makeCompressibleBuffer(bufferSize, i + 100);
    originalData[i].assign(buf->data(), buf->data() + buf->size());
    buffers.push_back(buf);
  }

  // Compress with parallel threads.
  auto compressResult = BlockPayload::fromBuffers(
      Payload::kCompressed, 300, std::move(buffers), &isValidity, pool_, codec_.get(), 3, threadPool_.get());
  ASSERT_TRUE(compressResult.ok());
  auto payload = std::move(*compressResult);

  // Serialize to bytes.
  auto bytes = serializePayload(payload);

  // Deserialize back.
  auto readCodecResult = arrow::util::Codec::Create(arrow::Compression::LZ4_FRAME);
  ASSERT_TRUE(readCodecResult.ok());
  auto readCodec = std::shared_ptr<arrow::util::Codec>(std::move(*readCodecResult));

  auto inputStream = std::make_shared<arrow::io::BufferReader>(bytes);
  uint32_t numRows = 0;
  int64_t deserializeTime = 0;
  int64_t decompressTime = 0;

  auto decompressResult = BlockPayload::deserialize(
      inputStream.get(), readCodec, pool_, numRows, deserializeTime, decompressTime);
  ASSERT_TRUE(decompressResult.ok());
  auto decompressedBuffers = std::move(*decompressResult);

  ASSERT_EQ(numRows, 300);
  ASSERT_EQ(decompressedBuffers.size(), numBuffers);

  // Verify each buffer matches original.
  for (int i = 0; i < numBuffers; i++) {
    ASSERT_NE(decompressedBuffers[i], nullptr);
    ASSERT_EQ(decompressedBuffers[i]->size(), static_cast<int64_t>(originalData[i].size()));
    EXPECT_EQ(memcmp(decompressedBuffers[i]->data(), originalData[i].data(), originalData[i].size()), 0)
        << "Buffer " << i << " mismatch after parallel compress -> decompress roundtrip";
  }
}

TEST_F(ParallelCompressionTest, ParallelWithManyThreads) {
  const int numBuffers = 16;
  std::vector<std::shared_ptr<arrow::Buffer>> buffers;
  std::vector<bool> isValidity(numBuffers, false);
  for (int i = 0; i < numBuffers; i++) {
    buffers.push_back(makeCompressibleBuffer(32 * 1024, i));
  }

  CompressionThreadPool pool8(8);
  auto result = BlockPayload::fromBuffers(
      Payload::kCompressed, 500, std::move(buffers), &isValidity, pool_, codec_.get(), 8, &pool8);
  ASSERT_TRUE(result.ok());
  auto payload = std::move(*result);
  ASSERT_NE(payload, nullptr);
}

TEST_F(ParallelCompressionTest, ParallelWithZstdCodec) {
  auto zstdResult = arrow::util::Codec::Create(arrow::Compression::ZSTD);
  ASSERT_TRUE(zstdResult.ok());
  auto zstdCodec = std::move(*zstdResult);

  const int numBuffers = 4;
  std::vector<bool> isValidity(numBuffers, false);

  std::vector<std::shared_ptr<arrow::Buffer>> buffers1;
  std::vector<std::shared_ptr<arrow::Buffer>> buffers2;
  for (int i = 0; i < numBuffers; i++) {
    auto original = makeCompressibleBuffer(64 * 1024, i + 200);
    auto copy = arrow::AllocateResizableBuffer(64 * 1024, pool_);
    ASSERT_TRUE(copy.ok());
    memcpy((*copy)->mutable_data(), original->data(), 64 * 1024);
    buffers1.push_back(original);
    buffers2.push_back(std::move(*copy));
  }

  auto result1 = BlockPayload::fromBuffers(
      Payload::kCompressed, 100, std::move(buffers1), &isValidity, pool_, zstdCodec.get(), 1);
  ASSERT_TRUE(result1.ok());
  auto payload1 = std::move(*result1);

  auto result2 = BlockPayload::fromBuffers(
      Payload::kCompressed, 100, std::move(buffers2), &isValidity, pool_, zstdCodec.get(), 4, threadPool_.get());
  ASSERT_TRUE(result2.ok());
  auto payload2 = std::move(*result2);

  auto bytes1 = serializePayload(payload1);
  auto bytes2 = serializePayload(payload2);

  ASSERT_EQ(bytes1->size(), bytes2->size());
  EXPECT_EQ(memcmp(bytes1->data(), bytes2->data(), bytes1->size()), 0);
}

} // namespace gluten
