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

#pragma once

#include "CudfVectorStream.h"
#include "compute/ResultIterator.h"
#include "memory/GpuBufferColumnarBatch.h"
#include "memory/VeloxColumnarBatch.h"
#include "velox/exec/Driver.h"
#include "velox/exec/Operator.h"
#include "velox/exec/Task.h"
#include "velox/experimental/cudf/exec/CudfOperator.h"
#include "velox/experimental/cudf/exec/Utilities.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

namespace gluten {

class CudfVectorStreamBase {
 public:
  virtual ~CudfVectorStreamBase() = default;

  explicit CudfVectorStreamBase(
      facebook::velox::exec::DriverCtx* driverCtx,
      facebook::velox::memory::MemoryPool* pool,
      ResultIterator* iterator,
      const facebook::velox::RowTypePtr& outputType)
      : driverCtx_(driverCtx), pool_(pool), outputType_(outputType), iterator_(iterator) {}

  bool hasNext();

  // Convert arrow batch to row vector, construct the new Rowvector with new outputType.
  virtual facebook::velox::RowVectorPtr next();

 protected:
  // Get the next batch from iterator_.
  std::shared_ptr<ColumnarBatch> nextInternal();

  facebook::velox::exec::DriverCtx* driverCtx_;
  facebook::velox::memory::MemoryPool* pool_;
  const facebook::velox::RowTypePtr outputType_;
  ResultIterator* iterator_;

  bool finished_{false};
};

class ValueStreamNode final : public facebook::velox::core::PlanNode {
 public:
  ValueStreamNode(
      const facebook::velox::core::PlanNodeId& id,
      const facebook::velox::RowTypePtr& outputType,
      std::shared_ptr<ResultIterator> iterator)
      : facebook::velox::core::PlanNode(id), outputType_(outputType), iterator_(std::move(iterator)) {}

  const facebook::velox::RowTypePtr& outputType() const override {
    return outputType_;
  }

  const std::vector<facebook::velox::core::PlanNodePtr>& sources() const override {
    return kEmptySources_;
  };

  ResultIterator* iterator() const {
    return iterator_.get();
  }

  std::string_view name() const override {
    return "ValueStream";
  }

  folly::dynamic serialize() const override {
    VELOX_UNSUPPORTED("ValueStream plan node is not serializable");
  }

 private:
  void addDetails(std::stringstream& stream) const override{};

  const facebook::velox::RowTypePtr outputType_;
  std::shared_ptr<ResultIterator> iterator_;
  const std::vector<facebook::velox::core::PlanNodePtr> kEmptySources_;
};

class CudfVectorStream : public CudfVectorStreamBase {
 public:
  static constexpr int32_t kDefaultTargetBatchRows = 100000;

  CudfVectorStream(
      facebook::velox::exec::DriverCtx* driverCtx,
      facebook::velox::memory::MemoryPool* pool,
      ResultIterator* iterator,
      const facebook::velox::RowTypePtr& outputType)
      : CudfVectorStreamBase(driverCtx, pool, iterator, outputType) {}

  bool hasPending() const {
    return !pendingRows_.empty() || stashedCudf_ != nullptr;
  }

  // Convert columnar batch to a CudfVector for downstream GPU operators.
  // Handles three input types with batch accumulation for Cases 2 & 3:
  //   1. VeloxColumnarBatch wrapping a CudfVector  -> re-wrap (returned immediately)
  //   2. VeloxColumnarBatch wrapping a CPU RowVector (e.g. BroadcastExchange)
  //      -> accumulated then batched upload
  //   3. GpuBufferColumnarBatch (shuffle read) -> accumulated then batched upload
  facebook::velox::RowVectorPtr next() override {
    // First, drain any pending accumulated rows.
    // Then accumulate more from the iterator until target batch size.
    while (pendingRowCount_ < kDefaultTargetBatchRows) {
      auto cb = nextInternal();
      if (cb == nullptr) {
        break;
      }

      // Case 1: VeloxColumnarBatch wrapping a CudfVector
      if (cb->getType() == "velox") {
        auto vb = std::dynamic_pointer_cast<VeloxColumnarBatch>(cb);
        VELOX_CHECK_NOT_NULL(vb);
        auto vp = vb->getRowVector();
        VELOX_CHECK_NOT_NULL(vp);
        auto cudfVector =
            std::dynamic_pointer_cast<facebook::velox::cudf_velox::CudfVector>(vp);
        if (cudfVector != nullptr) {
          // Already a CudfVector. If we have pending CPU rows, flush them
          // first and stash this CudfVector for the next call.
          if (!pendingRows_.empty()) {
            stashedCudf_ = cudfVector;
            break;
          }
          return std::make_shared<facebook::velox::cudf_velox::CudfVector>(
              vp->pool(), outputType_, vp->size(), cudfVector->release(), cudfVector->stream());
        }
        // Case 2: CPU RowVector – accumulate for batched upload.
        pendingRows_.push_back(vp);
        pendingRowCount_ += vp->size();
        continue;
      }

#ifdef GLUTEN_ENABLE_GPU
      // Case 3: GpuBufferColumnarBatch – convert to RowVector and accumulate.
      if (cb->getType() == "gpu") {
        auto gpuBatch = std::dynamic_pointer_cast<GpuBufferColumnarBatch>(cb);
        VELOX_CHECK_NOT_NULL(gpuBatch);
        auto rowVector = gpuBatch->toRowVector(pool_);
        VELOX_CHECK_NOT_NULL(rowVector);
        pendingRows_.push_back(rowVector);
        pendingRowCount_ += rowVector->size();
        continue;
      }
#endif
      VELOX_FAIL(
          "Unsupported ColumnarBatch type: '{}', numColumns: {}, numRows: {}",
          cb->getType(),
          cb->numColumns(),
          cb->numRows());
    }

    // If there's a stashed CudfVector and no pending CPU rows, return it.
    if (pendingRows_.empty() && stashedCudf_ != nullptr) {
      auto cudf = std::move(stashedCudf_);
      stashedCudf_ = nullptr;
      return std::make_shared<facebook::velox::cudf_velox::CudfVector>(
          cudf->pool(), outputType_, cudf->size(), cudf->release(), cudf->stream());
    }

    if (pendingRows_.empty()) {
      return nullptr;
    }

    // Batched HtoD: N async from_arrow, ONE sync, GPU concatenate.
    auto stream = facebook::velox::cudf_velox::cudfGlobalStreamPool().get_stream();
    auto tbl = facebook::velox::cudf_velox::with_arrow::toCudfTableBatched(
        pendingRows_, pool_, stream);
    VELOX_CHECK_NOT_NULL(tbl);
    const auto size = tbl->num_rows();

    pendingRows_.clear();
    pendingRowCount_ = 0;

    return std::make_shared<facebook::velox::cudf_velox::CudfVector>(
        pool_, outputType_, size, std::move(tbl), stream);
  }

 private:
  std::vector<facebook::velox::RowVectorPtr> pendingRows_;
  int32_t pendingRowCount_ = 0;
  std::shared_ptr<facebook::velox::cudf_velox::CudfVector> stashedCudf_;
};

// To avoid plan translator uses false node, this one cannot inherit ValueStreamNode.
class CudfValueStreamNode final : public facebook::velox::core::PlanNode {
 public:
  CudfValueStreamNode(
      const facebook::velox::core::PlanNodeId& id,
      const facebook::velox::RowTypePtr& outputType,
      std::shared_ptr<ResultIterator> iterator)
      : facebook::velox::core::PlanNode(id), outputType_(outputType), iterator_(std::move(iterator)) {}

  const facebook::velox::RowTypePtr& outputType() const override {
    return outputType_;
  }

  const std::vector<facebook::velox::core::PlanNodePtr>& sources() const override {
    return kEmptySources_;
  };

  ResultIterator* iterator() const {
    return iterator_.get();
  }

  std::string_view name() const override {
    return "CudfValueStream";
  }

  folly::dynamic serialize() const override {
    VELOX_UNSUPPORTED("CudfValueStream plan node is not serializable");
  }

 private:
  void addDetails(std::stringstream& stream) const override{};

  const facebook::velox::RowTypePtr outputType_;
  std::shared_ptr<ResultIterator> iterator_;
  const std::vector<facebook::velox::core::PlanNodePtr> kEmptySources_;
};

// Extends CudfOperator to identify it as GPU node, so not add CudfFormVelox operator.
class CudfValueStream : public facebook::velox::exec::SourceOperator, public facebook::velox::cudf_velox::CudfOperator {
 public:
  CudfValueStream(
      int32_t operatorId,
      facebook::velox::exec::DriverCtx* driverCtx,
      std::shared_ptr<const CudfValueStreamNode> valueStreamNode)
      : facebook::velox::exec::SourceOperator(
            driverCtx,
            valueStreamNode->outputType(),
            operatorId,
            valueStreamNode->id(),
            valueStreamNode->name().data()),
        facebook::velox::cudf_velox::CudfOperator(operatorId, valueStreamNode->id()) {
    ResultIterator* itr = valueStreamNode->iterator();
    rvStream_ = std::make_unique<CudfVectorStream>(driverCtx, pool(), itr, outputType_);
  }

  facebook::velox::RowVectorPtr getOutput() override {
    if (finished_) {
      return nullptr;
    }
    if (rvStream_->hasNext() || rvStream_->hasPending()) {
      auto result = rvStream_->next();
      if (result == nullptr) {
        finished_ = true;
      }
      return result;
    } else {
      finished_ = true;
      return nullptr;
    }
  }

  facebook::velox::exec::BlockingReason isBlocked(facebook::velox::ContinueFuture* /* unused */) override {
    return facebook::velox::exec::BlockingReason::kNotBlocked;
  }

  bool isFinished() override {
    return finished_;
  }

 private:
  bool finished_ = false;
  std::unique_ptr<CudfVectorStream> rvStream_;
};

class CudfVectorStreamOperatorTranslator : public facebook::velox::exec::Operator::PlanNodeTranslator {
  std::unique_ptr<facebook::velox::exec::Operator> toOperator(
      facebook::velox::exec::DriverCtx* ctx,
      int32_t id,
      const facebook::velox::core::PlanNodePtr& node) override {
    if (auto valueStreamNode = std::dynamic_pointer_cast<const CudfValueStreamNode>(node)) {
      return std::make_unique<CudfValueStream>(id, ctx, valueStreamNode);
    }
    return nullptr;
  }
};
} // namespace gluten
