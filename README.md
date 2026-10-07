# local-icp

本地点云 ICP（Iterative Closest Point）刚体配准工具。所有点云、变换矩阵和
测试数据只存在于本地文件或内存中，不依赖 ROS、PCL 配准模块或任何云/外部
服务。矩阵运算使用 Eigen（已 vendor 在 `third_party/eigen-3.4.0`），核心
ICP 流程为自行实现。

## 功能

- 支持 2D / 3D 点集（模板维度 `icp::Align<2>` / `icp::Align<3>`）。
- 通过最近邻（nearest-neighbor）建立对应关系，支持最大对应距离门限，
  明显离群点不参与优化。
- 基于 SVD 的闭式刚体估计：只产生旋转 + 平移，保证 `det(R) = +1`
  （无缩放、无 reflection）。
- 迭代执行 对应关系建立 → 刚体估计 → 点集更新，直到误差变化低于阈值
  或达到最大迭代次数。
- 返回最终 transform、RMSE、迭代次数、对应点数量和明确的收敛状态。
- 对空点云、共线/重合退化输入、严重不重叠点集返回明确的失败状态。

## 算法说明

### 对应关系（correspondence）

每次迭代中，先用当前估计 `(R, t)` 变换源点云，然后对每个变换后的源点
在目标点云中暴力搜索欧氏最近邻。距离超过
`Options::max_correspondence_distance` 的对应被拒绝（取负值表示不限制），
因此离群点和非重叠区域不会强制参与优化。

### SVD 刚体估计

给定对应点集 `{p_i}`（变换后的源点）和 `{q_i}`（目标点）：

1. 计算质心 `cp`、`cq`，将两组点去质心化。
2. 构造协方差矩阵 `H = Σ (p_i - cp)(q_i - cq)^T`。
3. 对 `H` 做 SVD：`H = U Σ V^T`。
4. `R = V · diag(1, …, 1, det(V U^T)) · U^T`。当 `det(V U^T) < 0` 时翻转
   最小奇异值对应的列，抑制 reflection，保证 `det(R) = +1`；不引入任何
   尺度因子（非 Umeyama 的相似变换版本）。
5. `t = cq - R · cp`。

估计出的增量变换左乘复合到当前 `(R, t)` 上，随后用内点对应计算 RMSE。

### 收敛条件

- 相邻两次迭代的 RMSE 变化 `|prev_rmse - rmse| < Options::tolerance`
  → `Status::kConverged`。
- 达到 `Options::max_iterations` 仍未满足上述条件
  → `Status::kMaxIterationsReached`（仍返回当前最优变换，`ok() == true`）。

### 失败状态

- `kInvalidInput`：点云为空或点数少于最小约束数（2D 需 2 点，3D 需 3 点）。
- `kDegenerateGeometry`：点云（或匹配到的对应点）重合/共线，去质心后的
  秩不足（2D 秩 < 1，3D 秩 < 2），刚体位姿不可观测。
- `kInsufficientCorrespondences`：在最大对应距离内找不到足够的对应点，
  点集很可能不重叠。

## API 摘要

```cpp
#include "icp/icp.hpp"

icp::PointCloud<3> source = ..., target = ...;
icp::Options options;
options.max_iterations = 60;
options.tolerance = 1e-9;
options.max_correspondence_distance = 1.0;  // 负值 = 不限制

icp::Result<3> result = icp::Align<3>(source, target, options);
// 或带初值：icp::Align<3>(source, target, options, R0, t0);

if (result.ok()) {
  // result.rotation, result.translation: target ~= R * source + t
  // result.rmse, result.iterations, result.num_correspondences
}
```

## 构建与测试

所有测试均在终端执行，测试点云由代码在内存中生成（固定随机种子，结果
可复现），覆盖：无噪声、含噪声、离群点、部分重叠、差初值、2D、空点云、
共线/重合退化、严重不重叠和迭代预算耗尽等场景，并校验旋转/平移误差
以及 `det(R) = +1`、`R` 正交（无缩放无反射）。

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/icp_tests          # 直接运行测试程序
ctest --test-dir build --output-on-failure   # 或通过 CTest 运行
```

## 目录结构

- `include/icp/icp.hpp` — 公开 API（`Options`、`Result`、`Status`、`Align`）。
- `src/icp.cpp` — ICP 主循环、最近邻对应、SVD 刚体估计、退化检测。
- `tests/test_icp.cpp` — 终端测试套件（无外部测试框架依赖）。
- `third_party/eigen-3.4.0` — vendored Eigen，仅用于矩阵运算。
