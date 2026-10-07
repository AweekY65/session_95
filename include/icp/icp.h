#pragma once

#include <string>

#include <Eigen/Dense>

namespace icp {

enum class IcpStatus {
  kConverged,                    // RMSE change dropped below tolerance.
  kMaxIterationsReached,         // Iteration budget exhausted before convergence.
  kEmptyInput,                   // Source or target cloud has no points.
  kDimensionMismatch,            // Clouds have different / unsupported dimensions.
  kDegenerateInput,              // Correspondences are collinear / coincident.
  kInsufficientCorrespondences,  // Too few inlier matches (severe non-overlap).
};

std::string ToString(IcpStatus status);

struct IcpOptions {
  int max_iterations = 50;
  // Convergence is declared when |prev_rmse - rmse| < tolerance.
  double tolerance = 1e-6;
  // Pairs farther apart than this are rejected as outliers.
  // A value <= 0 disables the distance gate.
  double max_correspondence_distance = -1.0;
  // Singular-value threshold used to detect degenerate (collinear /
  // coincident) correspondence sets.
  double degeneracy_epsilon = 1e-9;
};

struct IcpResult {
  IcpStatus status = IcpStatus::kConverged;
  // Final (D+1)x(D+1) homogeneous transform mapping source -> target.
  // Identity when the run failed before any estimate was produced.
  Eigen::MatrixXd transform;
  // Root-mean-square correspondence error after the final update.
  // -1 when no valid correspondence set was ever established.
  double rmse = -1.0;
  int iterations = 0;
  int num_correspondences = 0;
  std::string message;

  bool Success() const {
    return status == IcpStatus::kConverged ||
           status == IcpStatus::kMaxIterationsReached;
  }
};

// Point-to-point ICP for 2D or 3D point sets.
//
// Points are stored one per row (N x D, D in {2, 3}). Correspondences are
// brute-force nearest neighbors, the rigid increment is estimated with SVD
// (reflection-free, no scaling), and the transform is composed until the
// RMSE change falls below the tolerance or the iteration budget is spent.
class Icp {
 public:
  explicit Icp(IcpOptions options = IcpOptions{}) : options_(options) {}

  // `initial_transform` may be empty (identity is used) or a (D+1)x(D+1)
  // homogeneous matrix.
  IcpResult Align(const Eigen::MatrixXd& source,
                  const Eigen::MatrixXd& target,
                  const Eigen::MatrixXd& initial_transform = {}) const;

 private:
  IcpOptions options_;
};

}  // namespace icp
