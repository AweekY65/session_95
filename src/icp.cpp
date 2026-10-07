#include "icp/icp.hpp"

#include <cmath>
#include <limits>
#include <utility>

namespace icp {

std::string ToString(Status status) {
  switch (status) {
    case Status::kConverged:
      return "converged";
    case Status::kMaxIterationsReached:
      return "max_iterations_reached";
    case Status::kInvalidInput:
      return "invalid_input";
    case Status::kDegenerateGeometry:
      return "degenerate_geometry";
    case Status::kInsufficientCorrespondences:
      return "insufficient_correspondences";
  }
  return "unknown";
}

namespace {

// Minimum number of points needed to constrain a rigid transform in Dim
// dimensions: 2 for 2D, 3 for 3D.
template <int Dim>
constexpr int MinPoints() {
  return Dim;
}

// Rank required of the centered point set so the rigid pose is observable:
// 2D points must not all coincide (rank >= 1), 3D points must not be
// collinear (rank >= 2).
template <int Dim>
constexpr int MinRank() {
  return Dim - 1;
}

// Numerical rank of the centered cloud, used to detect coincident / collinear
// degeneracy. Returns -1 when the cloud is too small to ever be valid.
template <int Dim>
int CenteredRank(const PointCloud<Dim>& points) {
  if (static_cast<int>(points.size()) < MinPoints<Dim>()) {
    return -1;
  }
  Eigen::Matrix<double, Dim, Eigen::Dynamic> mat(Dim, points.size());
  Point<Dim> centroid = Point<Dim>::Zero();
  for (const auto& p : points) centroid += p;
  centroid /= static_cast<double>(points.size());
  for (size_t i = 0; i < points.size(); ++i) {
    mat.col(i) = points[i] - centroid;
  }
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(mat, Eigen::ComputeThinU | Eigen::ComputeThinV);
  const double max_sv = svd.singularValues().size() > 0 ? svd.singularValues()(0) : 0.0;
  const double threshold =
      std::numeric_limits<double>::epsilon() * Dim * std::max(max_sv, 1e-12) * 1e3;
  int rank = 0;
  for (int i = 0; i < svd.singularValues().size(); ++i) {
    if (svd.singularValues()(i) > threshold) ++rank;
  }
  return rank;
}

// Brute-force nearest neighbor. Returns (index, squared distance).
template <int Dim>
std::pair<int, double> NearestNeighbor(const Point<Dim>& query,
                                       const PointCloud<Dim>& target) {
  int best = -1;
  double best_dist = std::numeric_limits<double>::infinity();
  for (size_t i = 0; i < target.size(); ++i) {
    const double d = (query - target[i]).squaredNorm();
    if (d < best_dist) {
      best_dist = d;
      best = static_cast<int>(i);
    }
  }
  return {best, best_dist};
}

// Closed-form rigid transform (Umeyama/Horn without scale) mapping `from`
// onto `to` via SVD. Reflections are suppressed by correcting the smallest
// singular vector when det(V U^T) < 0, guaranteeing det(R) = +1.
template <int Dim>
bool EstimateRigid(const std::vector<Point<Dim>>& from, const std::vector<Point<Dim>>& to,
                   Eigen::Matrix<double, Dim, Dim>* rotation,
                   Eigen::Matrix<double, Dim, 1>* translation) {
  const size_t n = from.size();
  Point<Dim> centroid_from = Point<Dim>::Zero();
  Point<Dim> centroid_to = Point<Dim>::Zero();
  for (size_t i = 0; i < n; ++i) {
    centroid_from += from[i];
    centroid_to += to[i];
  }
  centroid_from /= static_cast<double>(n);
  centroid_to /= static_cast<double>(n);

  Eigen::Matrix<double, Dim, Dim> covariance = Eigen::Matrix<double, Dim, Dim>::Zero();
  for (size_t i = 0; i < n; ++i) {
    covariance += (from[i] - centroid_from) * (to[i] - centroid_to).transpose();
  }

  Eigen::JacobiSVD<Eigen::Matrix<double, Dim, Dim>> svd(
      covariance, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix<double, Dim, Dim> correction = Eigen::Matrix<double, Dim, Dim>::Identity();
  if ((svd.matrixV() * svd.matrixU().transpose()).determinant() < 0.0) {
    correction(Dim - 1, Dim - 1) = -1.0;
  }
  *rotation = svd.matrixV() * correction * svd.matrixU().transpose();
  *translation = centroid_to - (*rotation) * centroid_from;
  return true;
}

template <int Dim>
Result<Dim> RunIcp(const PointCloud<Dim>& source, const PointCloud<Dim>& target,
                   const Options& options,
                   const Eigen::Matrix<double, Dim, Dim>& initial_rotation,
                   const Eigen::Matrix<double, Dim, 1>& initial_translation) {
  using Matrix = Eigen::Matrix<double, Dim, Dim>;
  using Vector = Eigen::Matrix<double, Dim, 1>;

  Result<Dim> result;
  result.rotation = initial_rotation;
  result.translation = initial_translation;

  if (source.size() < static_cast<size_t>(MinPoints<Dim>()) ||
      target.size() < static_cast<size_t>(MinPoints<Dim>())) {
    result.status = Status::kInvalidInput;
    result.message = "point cloud is empty or has fewer than " +
                     std::to_string(MinPoints<Dim>()) + " points";
    return result;
  }
  if (CenteredRank(source) < MinRank<Dim>() || CenteredRank(target) < MinRank<Dim>()) {
    result.status = Status::kDegenerateGeometry;
    result.message = "points are coincident or collinear; rigid pose is unobservable";
    return result;
  }

  const double max_dist_sq =
      options.max_correspondence_distance >= 0.0
          ? options.max_correspondence_distance * options.max_correspondence_distance
          : std::numeric_limits<double>::infinity();

  double prev_rmse = std::numeric_limits<double>::infinity();
  for (int iter = 0; iter < options.max_iterations; ++iter) {
    // 1. Transform the source with the current estimate.
    PointCloud<Dim> transformed(source.size());
    for (size_t i = 0; i < source.size(); ++i) {
      transformed[i] = result.rotation * source[i] + result.translation;
    }

    // 2. Nearest-neighbor correspondences with an outlier distance gate.
    std::vector<Point<Dim>> corr_from;
    std::vector<Point<Dim>> corr_to;
    corr_from.reserve(source.size());
    corr_to.reserve(source.size());
    for (const auto& p : transformed) {
      const auto nn = NearestNeighbor(p, target);
      if (nn.second <= max_dist_sq) {
        corr_from.push_back(p);
        corr_to.push_back(target[nn.first]);
      }
    }

    result.num_correspondences = static_cast<int>(corr_from.size());
    if (static_cast<int>(corr_from.size()) < MinPoints<Dim>()) {
      result.status = Status::kInsufficientCorrespondences;
      result.message = "fewer than " + std::to_string(MinPoints<Dim>()) +
                       " correspondences within max_correspondence_distance; "
                       "clouds likely do not overlap";
      result.iterations = iter;
      return result;
    }
    if (CenteredRank(corr_from) < MinRank<Dim>() || CenteredRank(corr_to) < MinRank<Dim>()) {
      result.status = Status::kDegenerateGeometry;
      result.message = "matched correspondences are coincident or collinear";
      result.iterations = iter;
      return result;
    }

    // 3. Optimal rigid update from the current correspondences.
    Matrix delta_r;
    Vector delta_t;
    EstimateRigid(corr_from, corr_to, &delta_r, &delta_t);
    result.rotation = delta_r * result.rotation;
    result.translation = delta_r * result.translation + delta_t;

    // 4. RMSE of the inlier correspondences after the update.
    double sq_error = 0.0;
    for (size_t i = 0; i < corr_from.size(); ++i) {
      sq_error += (delta_r * corr_from[i] + delta_t - corr_to[i]).squaredNorm();
    }
    const double rmse = std::sqrt(sq_error / static_cast<double>(corr_from.size()));
    result.rmse = rmse;
    result.iterations = iter + 1;

    if (std::abs(prev_rmse - rmse) < options.tolerance) {
      result.status = Status::kConverged;
      result.message = "rmse change below tolerance";
      return result;
    }
    prev_rmse = rmse;
  }

  result.status = Status::kMaxIterationsReached;
  result.message = "iteration budget exhausted before rmse change fell below tolerance";
  return result;
}

}  // namespace

template <int Dim>
Result<Dim> Align(const PointCloud<Dim>& source, const PointCloud<Dim>& target,
                  const Options& options) {
  const Eigen::Matrix<double, Dim, Dim> identity =
      Eigen::Matrix<double, Dim, Dim>::Identity();
  const Eigen::Matrix<double, Dim, 1> zero = Eigen::Matrix<double, Dim, 1>::Zero();
  return RunIcp(source, target, options, identity, zero);
}

template <int Dim>
Result<Dim> Align(const PointCloud<Dim>& source, const PointCloud<Dim>& target,
                  const Options& options,
                  const Eigen::Matrix<double, Dim, Dim>& initial_rotation,
                  const Eigen::Matrix<double, Dim, 1>& initial_translation) {
  return RunIcp(source, target, options, initial_rotation, initial_translation);
}

template Result<2> Align<2>(const PointCloud<2>&, const PointCloud<2>&, const Options&);
template Result<3> Align<3>(const PointCloud<3>&, const PointCloud<3>&, const Options&);
template Result<2> Align<2>(const PointCloud<2>&, const PointCloud<2>&, const Options&,
                            const Eigen::Matrix2d&, const Eigen::Vector2d&);
template Result<3> Align<3>(const PointCloud<3>&, const PointCloud<3>&, const Options&,
                            const Eigen::Matrix3d&, const Eigen::Vector3d&);

}  // namespace icp
