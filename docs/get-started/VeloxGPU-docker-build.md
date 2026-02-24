---
layout: page
title: Velox GPU Docker Build Guide
nav_order: 10
parent: Getting-Started
---

# Gluten + Velox cuDF Build Guide

**Gluten branch:** `test_build`
**Velox branch:** `alfxu_dev` (NVIDIA internal: `gitlab-master.nvidia.com/alfxu/velox.git`)
**Default Docker image:** `apache/gluten:centos-9-jdk8-cudf`

---

## Part 1: Build Guide

### Prerequisites (host machine)

- Docker with `nvidia-container-toolkit` configured (see [`dev/start-cudf.sh`](../../../dev/start-cudf.sh) for host setup)
- NVIDIA driver compatible with CUDA 12.8+
- Both repos checked out side-by-side:
  ```
  velox_cudf_ws/
  ├── gluten/   (branch: test_build)
  └── velox/    (branch: alfxu_dev)
  ```

### Build

Run the build script from the **gluten repo root**:

```bash
bash dev/docker-build-cudf.sh
```

The script handles all steps automatically:
1. Starts (or reuses) the Docker container with GPU access and both repos bind-mounted
2. Verifies `nvidia-smi` is working inside the container
3. Runs `buildbundle-veloxbe.sh` (Arrow + Velox + Gluten C++ + Maven)
4. Runs `build-thirdparty.sh` (3rd-party JARs)

Logs are written to `build.log` and `thirdparty.log` in the gluten root. Monitor from the host:

```bash
tail -f build.log
```

**Key options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--system=` | `centos9` | Target OS image (`centos9` or `ubuntu2204`) |
| `--spark_version=` | `3.5` | Spark version (`3.3`, `3.4`, `3.5`, `4.0`, `4.1`, `ALL`) |
| `--cuda_arch=` | interactive | CUDA arch: `native`, `all-major`, or a specific SM number (e.g. `90`) |
| `--rebuild` | off | Incremental rebuild: skip Arrow, clear cmake cache, re-run velox + C++ + Maven |
| `--container=` | `gluten_cudf_build` | Docker container name |
| `--image=` | derived from `--system` | Override the Docker image explicitly |
| `--velox_repo=` | NVIDIA internal fork | Velox git URL (used only if velox dir is absent) |
| `--velox_branch=` | `alfxu_dev` | Velox branch to clone |

**Build stages (inside the container):**

1. **Arrow C++ + Java JNI** (~20–30 min) — downloads apache-arrow 15.0.0, applies Gluten patches, builds
2. **cuDF 26.04 from source** (~60–90 min) — `FetchContent` downloads commit `fc213fc1`, builds `libcudf.so` with CUDA
3. **Velox** (~30–60 min) — `make release` in `/opt/velox` with `VELOX_ENABLE_CUDF=ON`
4. **Gluten C++** (~10 min) — cmake + make in `gluten/cpp/`
5. **Maven JAR** (~5 min) — `mvn install -Pbackends-velox -Pspark-3.5 -DskipTests`

### Output artifacts

After a successful build, the Gluten JAR is at:

```
gluten/package/target/gluten-velox-bundle-spark3.5_2.12-linux_amd64-1.6.0-SNAPSHOT.jar
```

The native shared library (`libgluten.so`) is bundled inside the JAR.

---

## Part 2: Running with Spark

### Spark configuration

Enable Gluten by adding the JAR and plugin config to your Spark session:

```bash
export GLUTEN_JAR=<path-to-gluten>/package/target/gluten-velox-bundle-spark3.5_2.12-linux_amd64-1.6.0-SNAPSHOT.jar

spark-shell \
  --conf spark.plugins=org.apache.gluten.GlutenPlugin \
  --conf spark.driver.extraClassPath=${GLUTEN_JAR} \
  --conf spark.executor.extraClassPath=${GLUTEN_JAR} \
  --conf spark.memory.offHeap.enabled=true \
  --conf spark.memory.offHeap.size=20g \
  --conf spark.gluten.sql.columnar.forceShuffledHashJoin=true \
  --conf spark.shuffle.manager=org.apache.spark.shuffle.sort.ColumnarShuffleManager
```

### Enable GPU (cuDF) acceleration

Add these confs on top of the base Gluten config:

```bash
  --conf spark.gluten.sql.columnar.cudf=true \
  --conf spark.gluten.debug.enabled.cudf=true   # optional: logs which ops go to GPU
```

> **Note:** When `spark.gluten.sql.columnar.cudf=true`, Velox PlanNodes are converted to GPU
> operators. Unsupported operators fall back to CPU automatically.

### GPU operator support status

| Operator | Status | Notes |
|----------|--------|-------|
| Scan | Not supported | In development |
| Filter | Implemented | |
| Project | Partial | TPC-H compatible |
| Aggregation | Partial | TPC-H compatible |
| Join | Partial | TPC-H compatible |
| OrderBy | Implemented | |
| Spill | Not supported | In planning |

### Running tests inside the container

The container has Java 8 and Maven pre-installed. To run a quick Spark test:

```bash
docker exec -it gluten_cudf_build bash
# inside container:
cd /opt/gluten
./build/mvn -pl backends-velox test -Pbackends-velox -Pspark-3.5 \
  -Dtest=<TestClassName> -DfailIfNoTests=false
```

---

## Part 3: Debugging — cuDF Version Mismatch

### Error

Build fails at a velox compile step with:

```
/opt/velox/velox/experimental/cudf/exec/CudfHashAggregation.cpp:35:
fatal error: cudf/reduction/approx_distinct_count.hpp: No such file or directory
```

### Root cause

The Docker image has a **partial RAPIDS upgrade** — rmm was updated to 26.04 but cudf itself is still 26.02:

```
-- Setting cudf source to AUTO
-- [cudf] Found rmm: 26.04.0      ← rmm is 26.04
-- [cudf] Using SYSTEM cudf       ← but system cudf is 26.02!
```

| Component | Version |
|-----------|---------|
| Docker image cudf | **26.02** (`/usr/local/lib64/cmake/cudf`) |
| velox `alfxu_dev` requires | **26.04** (`CMake/resolve_dependency_modules/cudf.cmake`) |
| Target cudf commit | `fc213fc1ad889e2edf291b5555764ce677cb5dfa` (2026-02-10) |

`cudf::approx_distinct_count` (header + compiled symbols) was added in 26.04 and is entirely absent from 26.02.

### Fix

In `ep/build-velox/src/build-velox.sh`, add `-Dcudf_SOURCE=BUNDLED` to the GPU cmake options:

```diff
  if [ $ENABLE_GPU == "ON" ]; then
      echo "enable GPU support."
      COMPILE_OPTION="$COMPILE_OPTION -DVELOX_ENABLE_GPU=ON -DVELOX_ENABLE_CUDF=ON -DCMAKE_CUDA_ARCHITECTURES=75 \
-         -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc"
+         -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -Dcudf_SOURCE=BUNDLED"
  fi
```

This forces `FetchContent_MakeAvailable(cudf)` to download and build cudf 26.04 from source instead of picking up the stale system cudf 26.02.

After applying the fix, use the `--rebuild` flag to resume without re-running Arrow:

```bash
bash dev/docker-build-cudf.sh --rebuild
```

`--rebuild` automatically clears the stale CMakeCache and re-runs only the velox + Gluten C++ + Maven steps.

---

## Notes

- The `ep/build-velox/src/build-velox.sh` change (`-Dcudf_SOURCE=BUNDLED`) should be committed to `test_build` so future GPU builds against velox 26.04+ don't hit the same issue. It can be removed once the Docker image is rebuilt with cudf 26.04 pre-installed.
- ccache is available in the container (`/usr/bin/ccache`) and speeds up incremental rebuilds significantly.
- Build logs are written to `gluten/build.log` (overwritten on full builds, appended with `--rebuild`).
