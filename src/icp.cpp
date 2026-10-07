#include "icp/icp.h"

#include <Eigen/SVD>

#include <cmath>
#include <limits>
#include <vector>

namespace icp {

std::string ToString(IcpStatus status) {
  switch (status) {
    case IcpStatus::kConverged:
      return "Converged";
    case IcpStatus::kMaxIterationsReached:
      return "MaxIterationsReached";
    case IcpStatus::kEmptyInput:
      return "EmptyInput";
    case IcpStatus::kDimensionMismatch:
      return "DimensionMismatch";
    case IcpStatus::kDegenerateInput:
      return "DegenerateInput";
    case IcpStatus::kInsufficientCorrespondences:
      return "InsufficientCorrespondences";
  }
  return "Unknown";
}

namespace {

// Brute-force nearest-neighbor correspondences between the (already
// transformed) source points and the target cloud. Pairs whose distance
// exceeds `max_dist` (when > 0) are rejected as outliers.
struct CorrespondenceSet {
  std::vector<int> src_indices;
  std::vector<int> dst_indices;
};

CorrespondenceSet FindCorrespondences(const Eigen::MatrixXd& src_transformed,
                                      const Eigen::MatrixXd& target,
                                      double max_dist) {
  CorrespondenceSet pairs;
  const double max_dist_sq = max_dist * max_dist;
  for (int i = 0; i < src_transformed.rows(); ++i) {
    double best_dist_sq = std::numeric_limits<double>::max();
    int best_j = -1;
    for (int j = 0; j < target.rows(); ++j) {
      const double dist_sq = (src_transformed.row(i) - target.row(j)).squaredNorm();
      if (dist_sq < best_dist_sq) {
        best_dist_sq = dist_sq;
        best_j = j;
      }
    }
    if (best_j < 0) continue;
    if (max_dist > 0.0 && best_dist_sq > max_dist_sq) continue;
    pairs.src_indices.push_back(i);
    pairs.dst_indices.push_back(best_j);
  }
  return pairs;
}

Eigen::MatrixXd GatherRows(const Eigen::MatrixXd& points,
                           const std::vector<int>& indices) {
  Eigen::MatrixXd out(static_cast<int>(indices.size()), points.cols());
  for (size_t k = 0; k < indices.size(); ++k) {
    out.row(static_cast<int>(k)) = points.row(indices[k]);
  }
  return out;
}

// Returns true when the matched point set is collinear or coincident, i.e.
// the centered covariance has rank < 2 (the second-largest eigenvalue is
// negligible relative to the largest).
bool IsDegenerate(const Eigen::MatrixXd& centered, double degeneracy_epsilon) {
  const Eigen::MatrixXd cov =
      centered.transpose() * centered / static_cast<double>(centered.rows());
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(cov);
  const Eigen::VectorXd eigenvalues = solver.eigenvalues();  // ascending
  const int dim = static_cast<int>(centered.cols());
  const double largest = eigenvalues(dim - 1);
  const double second = eigenvalues(dim - 2);
  return second <= degeneracy_epsilon * std::max(largest, 0.0);
}

// Kabsch / Umeyama rigid estimation (no scaling, reflection corrected).
// Computes the homogeneous transform mapping `src` onto `dst`.
bool EstimateRigidTransform(const Eigen::MatrixXd& src,
                            const Eigen::MatrixXd& dst,
                            double degeneracy_epsilon,
                            Eigen::MatrixXd* transform) {
  const int dim = static_cast<int>(src.cols());
  const Eigen::VectorXd mu_src = src.colwise().mean();
  const Eigen::VectorXd mu_dst = dst.colwise().mean();
  const Eigen::MatrixXd p = src.rowwise() - mu_src.transpose();
  const Eigen::MatrixXd q = dst.rowwise() - mu_dst.transpose();

  if (IsDegenerate(p, degeneracy_epsilon)) return false;

  const Eigen::MatrixXd h = p.transpose() * q;
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(h, Eigen::ComputeFullU | Eigen::ComputeFullV);
  const Eigen::MatrixXd u = svd.matrixU();
  const Eigen::MatrixXd v = svd.matrixV();

  // Force a proper rotation: det(R) = +1, never a reflection.
  Eigen::MatrixXd d = Eigen::MatrixXd::Identity(dim, dim);
  d(dim - 1, dim - 1) = (v * u.transpose()).determinant() < 0.0 ? -1.0 : 1.0;
  const Eigen::MatrixXd r = v * d * u.transpose();
  const Eigen::VectorXd t = mu_dst - r * mu_src;

  transform->setIdentity(dim + 1, dim + 1);
  transform->block(0, 0, dim, dim) = r;
  transform->block(0, dim, dim, 1) = t;
  return true;
}

Eigen::MatrixXd ApplyTransform(const Eigen::MatrixXd& points,
                               const Eigen::MatrixXd& transform) {
  const int dim = static_cast<int>(points.cols());
  const Eigen::VectorXd t = transform.block(0, dim, dim, 1);
  return ((transform.block(0, 0, dim, dim) * points.transpose()).colwise() + t)
      .transpose();
}

double ComputeRmse(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b) {
  return std::sqrt((a - b).rowwise().squaredNorm().mean());
}

}  // namespace

IcpResult Icp::Align(const Eigen::MatrixXd& source, const Eigen::MatrixXd& target,
                     const Eigen::MatrixXd& initial_transform) const {
  IcpResult result;

  if (source.rows() == 0 || target.rows() == 0) {
    result.status = IcpStatus::kEmptyInput;
    result.message = "Source or target point cloud is empty.";
    result.transform = Eigen::MatrixXd{};
    return result;
  }
  const int dim = static_cast<int>(source.cols());
  if (target.cols() != dim || (dim != 2 && dim != 3)) {
    result.status = IcpStatus::kDimensionMismatch;
    result.message = "Point clouds must share a supported dimension (2 or 3).";
    result.transform = Eigen::MatrixXd{};
    return result;
  }

  Eigen::MatrixXd transform = Eigen::MatrixXd::Identity(dim + 1, dim + 1);
  if (initial_transform.size() != 0) {
    if (initial_transform.rows() != dim + 1 || initial_transform.cols() != dim + 1) {
      result.status = IcpStatus::kDimensionMismatch;
      result.message = "Initial transform must be (D+1)x(D+1).";
      result.transform = Eigen::MatrixXd{};
      return result;
    }
    transform = initial_transform;
  }
  result.transform = transform;

  // A rigid transform in D dimensions needs at least D+1 non-degenerate
  // correspondences to be observable.
  const size_t min_correspondences = static_cast<size_t>(dim) + 1;

  double prev_rmse = std::numeric_limits<double>::max();
  for (int iter = 0; iter < options_.max_iterations; ++iter) {
    const Eigen::MatrixXd src_transformed = ApplyTransform(source, transform);
    const CorrespondenceSet pairs = FindCorrespondences(
        src_transformed, target, options_.max_correspondence_distance);

    if (pairs.src_indices.size() < min_correspondences) {
      result.status = IcpStatus::kInsufficientCorrespondences;
      result.message =
          "Too few inlier correspondences (" +
          std::to_string(pairs.src_indices.size()) + " of " +
          std::to_string(min_correspondences) +
          " required); the clouds likely do not overlap.";
      result.iterations = iter;
      result.num_correspondences = static_cast<int>(pairs.src_indices.size());
      return result;
    }

    const Eigen::MatrixXd matched_src = GatherRows(src_transformed, pairs.src_indices);
    const Eigen::MatrixXd matched_dst = GatherRows(target, pairs.dst_indices);

    Eigen::MatrixXd delta;
    if (!EstimateRigidTransform(matched_src, matched_dst,
                                options_.degeneracy_epsilon, &delta)) {
      result.status = IcpStatus::kDegenerateInput;
      result.message =
          "Correspondence set is collinear or coincident; the rigid "
          "transform is not observable.";
      result.iterations = iter;
      result.num_correspondences = static_cast<int>(pairs.src_indices.size());
      return result;
    }

    transform = delta * transform;
    result.transform = transform;
    result.iterations = iter + 1;
    result.num_correspondences = static_cast<int>(pairs.src_indices.size());

    const double rmse = ComputeRmse(ApplyTransform(matched_src, delta), matched_dst);
    result.rmse = rmse;

    if (std::abs(prev_rmse - rmse) < options_.tolerance) {
      result.status = IcpStatus::kConverged;
      result.message = "RMSE change dropped below tolerance.";
      return result;
    }
    prev_rmse = rmse;
  }

  result.status = IcpStatus::kMaxIterationsReached;
  result.message = "Iteration budget exhausted before convergence.";
  return result;
}

}  // namespace icp
