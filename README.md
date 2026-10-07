# local-icp

A self-contained, local-only point-cloud ICP (Iterative Closest Point) rigid
registration tool. All point clouds, transforms, and test data live in local
files or in memory. There is **no** dependency on ROS, PCL registration
modules, cloud vision services, or any external service. Eigen (vendored in
`third_party/eigen`) is used only for matrix math; the full ICP pipeline is
implemented in this repository.

## Algorithm

Points are stored one per row in an `N x D` matrix, with `D` in `{2, 3}`.

1. **Correspondence** — for every transformed source point, the nearest
   target point is found by brute-force search. Pairs whose distance exceeds
   `max_correspondence_distance` (when > 0) are rejected, so gross outliers
   and non-overlapping regions are never forced into the optimization.
2. **Rigid transform estimation (SVD / Kabsch)** — given the matched pairs,
   both sets are centered and the cross-covariance `H = P^T Q` is decomposed
   with SVD (`H = U S V^T`). The rotation is `R = V * diag(1, ..., 1, det(V U^T)) * U^T`,
   which guarantees `det(R) = +1`: no reflection and no scaling is ever
   produced. The translation is `t = mu_target - R * mu_source`.
3. **Update** — the incremental transform is composed onto the running
   estimate and the source cloud is re-transformed.
4. **Convergence** — steps 1-3 repeat until the absolute RMSE change between
   iterations drops below `tolerance` (status `Converged`) or
   `max_iterations` is exhausted (status `MaxIterationsReached`).

### Failure handling

- Empty source or target cloud -> `EmptyInput`.
- Mismatched or unsupported dimensions (anything but 2D/3D) ->
  `DimensionMismatch`.
- Collinear or coincident correspondence sets (rotation unobservable) ->
  `DegenerateInput`, detected via the eigenvalue spectrum of the centered
  covariance.
- Fewer inlier correspondences than required after distance gating (severe
  non-overlap) -> `InsufficientCorrespondences`.

`IcpResult` always reports the final homogeneous transform, RMSE, iteration
count, inlier correspondence count, and an explicit status.

## Build

```sh
cmake -S . -B build
cmake --build build -j
```

## Run the tests

```sh
ctest --test-dir build --output-on-failure
# or directly:
./build/icp_test
```

The tests synthesize point clouds with known rotation/translation in memory
and cover: exact 2D/3D recovery, Gaussian noise, outliers, partial overlap,
poor initial guesses, empty input, dimension mismatch, collinear/coincident
degeneracy, and severe non-overlap. Every successful case also verifies that
the estimated rotation stays orthonormal with determinant +1.

## CLI

```sh
./build/icp_cli source.txt target.txt [--init init.txt] [--out T.txt] \
    [--max-dist 0.5] [--tol 1e-6] [--iters 50]
```

Cloud files are plain ASCII, one point per row (2 or 3 coordinates). The tool
prints the status, RMSE, iteration count, and final transform, and optionally
writes the transform to a local file. Exit code is 0 on success.
