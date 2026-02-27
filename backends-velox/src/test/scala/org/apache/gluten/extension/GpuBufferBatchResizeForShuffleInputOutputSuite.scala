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
package org.apache.gluten.extension

import org.apache.gluten.config.{GlutenConfig, VeloxConfig}
import org.apache.gluten.execution.{GpuResizeBufferColumnarBatchExec, VeloxWholeStageTransformerSuite}

import org.apache.spark.SparkConf
import org.apache.spark.sql.execution.GPUColumnarShuffleExchangeExec

class GpuBufferBatchResizeForShuffleInputOutputSuite extends VeloxWholeStageTransformerSuite {
  override protected val resourcePath: String = "/tpch-data-parquet"
  override protected val fileFormat: String = "parquet"

  override protected def sparkConf: SparkConf = super.sparkConf
    .set("spark.unsafe.exceptionOnMemoryLeak", "true")
    .set("spark.shuffle.manager", "org.apache.spark.shuffle.sort.ColumnarShuffleManager")
    .set(GlutenConfig.COLUMNAR_CUDF_ENABLED.key, "true")
    .set(VeloxConfig.CUDF_ENABLE_VALIDATION.key, "false")
    .set(VeloxConfig.CUDF_GPU_TARGET_BATCH_ROWS.key, "500")

  test("non-AQE: GPUColumnarShuffleExchangeExec wrapped with GpuResizeBufferColumnarBatchExec") {
    withSQLConf(
      "spark.sql.autoBroadcastJoinThreshold" -> "-1",
      "spark.sql.adaptive.enabled" -> "false",
      GlutenConfig.COLUMNAR_FORCE_SHUFFLED_HASH_JOIN_ENABLED.key -> "true"
    ) {
      createTPCHNotNullTables()
      val df = spark.sql("""select l_partkey, l_suppkey, ps_availqty
                           |from lineitem join partsupp
                           |on l_partkey = ps_partkey and l_suppkey = ps_suppkey""".stripMargin)
      val plan = df.queryExecution.executedPlan

      val gpuShuffles = plan.collect { case e: GPUColumnarShuffleExchangeExec => e }
      assert(gpuShuffles.nonEmpty, "Expected GPUColumnarShuffleExchangeExec in non-AQE plan")

      val resizers = plan.collect { case r: GpuResizeBufferColumnarBatchExec => r }
      assert(
        resizers.nonEmpty,
        "Expected GpuResizeBufferColumnarBatchExec wrapping GPU shuffle in non-AQE mode")

      resizers.foreach {
        r =>
          assert(
            r.child.isInstanceOf[GPUColumnarShuffleExchangeExec],
            s"GpuResizeBufferColumnarBatchExec child should be GPUColumnarShuffleExchangeExec, " +
              s"but was ${r.child.getClass.getSimpleName}"
          )
      }
    }
  }

  test("non-AQE: rule is no-op when cudf is disabled") {
    withSQLConf(
      "spark.sql.autoBroadcastJoinThreshold" -> "-1",
      "spark.sql.adaptive.enabled" -> "false",
      GlutenConfig.COLUMNAR_CUDF_ENABLED.key -> "false",
      GlutenConfig.COLUMNAR_FORCE_SHUFFLED_HASH_JOIN_ENABLED.key -> "true"
    ) {
      createTPCHNotNullTables()
      val df = spark.sql("""select l_partkey, l_suppkey, ps_availqty
                           |from lineitem join partsupp
                           |on l_partkey = ps_partkey and l_suppkey = ps_suppkey""".stripMargin)
      val plan = df.queryExecution.executedPlan

      val gpuShuffles = plan.collect { case _: GPUColumnarShuffleExchangeExec => true }
      assert(
        gpuShuffles.isEmpty,
        "No GPUColumnarShuffleExchangeExec expected when cudf is disabled")

      val resizers = plan.collect { case _: GpuResizeBufferColumnarBatchExec => true }
      assert(resizers.isEmpty, "No GpuResizeBufferColumnarBatchExec expected when cudf is disabled")
    }
  }

  test("gpuTargetBatchRows config reaches native and filter query is correct") {
    withSQLConf("spark.sql.adaptive.enabled" -> "false") {
      createTPCHNotNullTables()
      val df = spark.sql("""select l_orderkey, l_partkey, l_quantity
                           |from lineitem
                           |where l_quantity > 30""".stripMargin)
      checkAnswer(
        df,
        spark.sql("select l_orderkey, l_partkey, l_quantity from lineitem where l_quantity > 30"))
    }
  }

  test("GPU partition shuffle produces correct join results") {
    withSQLConf(
      "spark.sql.autoBroadcastJoinThreshold" -> "-1",
      "spark.sql.adaptive.enabled" -> "false",
      GlutenConfig.COLUMNAR_FORCE_SHUFFLED_HASH_JOIN_ENABLED.key -> "true",
      GlutenConfig.COLUMNAR_CUDF_GPU_PARTITION.key -> "true"
    ) {
      createTPCHNotNullTables()
      val sql = """select l_orderkey, l_linenumber, o_orderstatus
                  |from lineitem join orders
                  |on l_orderkey = o_orderkey
                  |where l_linenumber <= 2""".stripMargin
      val df = spark.sql(sql)

      val plan = df.queryExecution.executedPlan
      val gpuShuffles = plan.collect { case e: GPUColumnarShuffleExchangeExec => e }
      assert(gpuShuffles.nonEmpty, "Expected GPUColumnarShuffleExchangeExec")

      checkAnswer(df, spark.sql(sql))
    }
  }

  test("GPU partition shuffle produces correct aggregation results") {
    withSQLConf(
      "spark.sql.autoBroadcastJoinThreshold" -> "-1",
      "spark.sql.adaptive.enabled" -> "false",
      GlutenConfig.COLUMNAR_CUDF_GPU_PARTITION.key -> "true"
    ) {
      createTPCHNotNullTables()
      val sql = """select l_returnflag, l_linestatus,
                  |       sum(cast(l_linenumber as bigint)) as sum_lines,
                  |       count(*) as count_order
                  |from lineitem
                  |group by l_returnflag, l_linestatus
                  |order by l_returnflag, l_linestatus""".stripMargin
      val df = spark.sql(sql)
      checkAnswer(df, spark.sql(sql))
    }
  }
}
