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

## TL;DR

Everything is driven by one script from the **gluten repo root**:

```bash
# First-time full build (interactive CUDA arch prompt)
bash dev/docker-build-cudf.sh

# Full build with explicit CUDA arch (non-interactive)
bash dev/docker-build-cudf.sh --cuda_arch=90

# Incremental rebuild after a code change (skip Arrow, clear cmake cache)
bash dev/docker-build-cudf.sh --rebuild

# Build for Ubuntu instead of the default CentOS 9
bash dev/docker-build-cudf.sh --system=ubuntu2204

# See all options
bash dev/docker-build-cudf.sh --help
```

The script takes care of everything: container lifecycle, GPU verification, cmake cache management, bundle build, and 3rd-party JARs. Logs land in `build.log` and `thirdparty.log` at the gluten root.

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

### Running the build

```bash
bash dev/docker-build-cudf.sh
```

On the first run, the script will prompt you to select a CUDA target architecture (it auto-detects the local GPU as a hint). Pass `--cuda_arch=` to skip the prompt:

```bash
bash dev/docker-build-cudf.sh --cuda_arch=native      # local GPU only
bash dev/docker-build-cudf.sh --cuda_arch=all-major   # sm_70,75,80,86,89,90
bash dev/docker-build-cudf.sh --cuda_arch=90          # pin to a specific SM
```

Monitor progress in a separate terminal:

```bash
tail -f build.log
```

### What the script does (5 steps)

| Step | Action |
|------|--------|
| 1 | Start (or reuse/restart) the Docker container with `--gpus all` and both repos bind-mounted |
| 2 | Verify GPU access via `nvidia-smi` inside the container |
| 3 | On `--rebuild`: clear `/opt/velox/_build/release/CMakeCache.txt`; otherwise no-op |
| 4 | Run `buildbundle-veloxbe.sh` — Arrow → cuDF → Velox → Gluten C++ → Maven JAR |
| 5 | Run `build-thirdparty.sh` — 3rd-party JARs |

**Build stage durations (step 4):**

| Stage | Time |
|-------|------|
| Arrow C++ + Java JNI | ~20–30 min |
| cuDF 26.04 from source (`fc213fc1`) | ~60–90 min |
| Velox (`VELOX_ENABLE_CUDF=ON`) | ~30–60 min |
| Gluten C++ | ~10 min |
| Maven JAR | ~5 min |

### All options

| Option | Default | Description |
|--------|---------|-------------|
| `--system=` | `centos9` | Target OS image (`centos9` or `ubuntu2204`) |
| `--spark_version=` | `3.5` | Spark version (`3.3`, `3.4`, `3.5`, `4.0`, `4.1`, `ALL`) |
| `--cuda_arch=` | interactive | `native`, `all-major`, or SM number (e.g. `90`) |
| `--rebuild` | off | Skip Arrow, clear cmake cache, re-run velox + C++ + Maven |
| `--container=` | `gluten_cudf_build` | Docker container name |
| `--image=` | derived from `--system` | Override the Docker image explicitly |
| `--velox_repo=` | NVIDIA internal fork | Velox git URL (used only if velox dir is absent) |
| `--velox_branch=` | `alfxu_dev` | Velox branch to clone |

### Output

```
gluten/package/target/gluten-velox-bundle-spark3.5_2.12-linux_amd64-1.6.0-SNAPSHOT.jar
```

`libgluten.so` (with `libvelox.so` and `libcudf.so`) is bundled inside the JAR.

---

## Part 2: Running with Spark

### Spark configuration

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

Add on top of the base config:

```bash
  --conf spark.gluten.sql.columnar.cudf=true \
  --conf spark.gluten.debug.enabled.cudf=true   # optional: logs which ops go to GPU
```

> When `spark.gluten.sql.columnar.cudf=true`, Velox PlanNodes are converted to GPU operators.
> Unsupported operators fall back to CPU automatically.

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

### Fix (commit `5afbccb`)

Three files are patched together in commit `5afbccb46ee5dec8fb2e53a28265f4857f1c9dd1`:

**1. `ep/build-velox/src/build-velox.sh` — force cudf 26.04 from source + dynamic CUDA arch**

```diff
+CUDA_ARCH="native"
 ...
+  --cuda_arch=*)
+    CUDA_ARCH=("${arg#*=}")
 ...
   if [ $ENABLE_GPU == "ON" ]; then
-    COMPILE_OPTION="$COMPILE_OPTION -DVELOX_ENABLE_GPU=ON -DVELOX_ENABLE_CUDF=ON -DCMAKE_CUDA_ARCHITECTURES=75 \
-        -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc"
+    COMPILE_OPTION="$COMPILE_OPTION -DVELOX_ENABLE_GPU=ON -DVELOX_ENABLE_CUDF=ON -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} \
+        -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -Dcudf_SOURCE=BUNDLED"
   fi
```

- `-Dcudf_SOURCE=BUNDLED` overrides cmake's `AUTO` resolution and forces `FetchContent` to download and build cudf 26.04 from source, bypassing the stale system cudf 26.02.
- `-DCMAKE_CUDA_ARCHITECTURES=75` was hardcoded (sm_75 = Turing); it is now `${CUDA_ARCH}` and driven by `--cuda_arch=` passed from `docker-build-cudf.sh`.

**2. `dev/builddeps-veloxbe.sh` — thread `--cuda_arch` through + guard velox clone**

```diff
+CUDA_ARCH="native"
 ...
+        --cuda_arch=*)
+        CUDA_ARCH=("${arg#*=}")
 ...
-  ./build-velox.sh ... --enable_gpu=$ENABLE_GPU --build_test_utils=$BUILD_TESTS \
+  ./build-velox.sh ... --enable_gpu=$ENABLE_GPU --cuda_arch=$CUDA_ARCH --build_test_utils=$BUILD_TESTS \
 ...
-    get_velox
+    if [ ! -d "$VELOX_HOME" ]; then
+      get_velox
+    else
+      echo "VELOX_HOME=$VELOX_HOME already exists, skipping Velox checkout."
+    fi
```

- Threads `--cuda_arch` all the way from `docker-build-cudf.sh` → `buildbundle-veloxbe.sh` → `builddeps-veloxbe.sh` → `build-velox.sh`.
- Guards `get_velox` so a mounted local velox dir (e.g. `/opt/velox`) is never overwritten by a fresh clone.

**3. `ep/build-velox/src/get-velox.sh` — switch default repo from IBM fork to upstream**

```diff
-VELOX_REPO=https://github.com/IBM/velox.git
-VELOX_BRANCH=dft-2026_02_06
-VELOX_ENHANCED_BRANCH=ibm-2026_02_06
+VELOX_REPO=https://github.com/facebookincubator/velox.git
+VELOX_BRANCH=main
+VELOX_ENHANCED_BRANCH=main
```

Restores the upstream Facebook velox as the default clone target. This only matters when `VELOX_HOME` does not already exist; since `docker-build-cudf.sh` always mounts the local velox dir, this path is normally skipped.

---

After applying all three patches, resume with:

```bash
bash dev/docker-build-cudf.sh --rebuild
```

`--rebuild` clears the stale `CMakeCache.txt` and re-runs only velox + Gluten C++ + Maven — no need to redo the ~30 min Arrow build.

---

## Notes

- The `-Dcudf_SOURCE=BUNDLED` change should be committed to `test_build`. It can be dropped once the Docker image ships cudf 26.04 pre-installed.
- ccache is available in the container (`/usr/bin/ccache`) and speeds up incremental rebuilds significantly.
- `build.log` is overwritten on full builds and appended on `--rebuild` runs.
