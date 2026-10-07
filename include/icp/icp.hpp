#pragma once

#include <Eigen/Dense>

#include <string>
#include <vector>

namespace icp {

// 2D or 3D point, stored as a fixed-size Eigen vector.
template <int Dim>
using Point = Eigen::Matrix<double, Dim, 1>;

template <int Dim>
using PointCloud = std::vector<Point<Dim>>;

enum class Status {
  kConverged,                    // RMSE change dropped below the tolerance.
  kMaxIterationsReached,         // Iteration budget exhausted before convergence.
  kInvalidInput,                 // Empty or too-small point clouds.
  kDegenerateGeometry,           // Coincident / collinear points make the pose unobservable.
  kInsufficientCorrespondences,  // Clouds do not overlap within max_correspondence_distance.
};

std::string ToString(Status status);

struct Options {
  int max_iterations = 60;
  // Converged when |prev_rmse - rmse| < tolerance.
  double tolerance = 1e-9;
  // Correspondences farther than this are rejected as outliers.
  // A negative value disables the gate (every nearest neighbor is kept).
  double max_correspondence_distance = -1.0;
};

template <int Dim>
struct Result {
  using Matrix = Eigen::Matrix<double, Dim, Dim>;
  using Vector = Eigen::Matrix<double, Dim, 1>;

  // Final rigid transform mapping source onto target: target ~= R * source + t.
  Matrix rotation = Matrix::Identity();
  Vector translation = Vector::Zero();

  double rmse = 0.0;             // RMSE of the final inlier correspondences.
  int iterations = 0;            // Iterations actually executed.
  int num_correspondences = 0;   // Inlier correspondences used in the last step.
  Status status = Status::kInvalidInput;
  std::string message;

  // True when a usable transform was produced (converged or hit max iterations).
  bool ok() const {
    return status == Status::kConverged || status == Status::kMaxIterationsReached;
  }
};

// Runs point-to-point ICP with an identity initial guess.
template <int Dim>
Result<Dim> Align(const PointCloud<Dim>& source, const PointCloud<Dim>& target,
                  const Options& options = Options{});

// Runs point-to-point ICP starting from a caller-provided initial guess.
template <int Dim>
Result<Dim> Align(const PointCloud<Dim>& source, const PointCloud<Dim>& target,
                  const Options& options,
                  const Eigen::Matrix<double, Dim, Dim>& initial_rotation,
                  const Eigen::Matrix<double, Dim, 1>& initial_translation);

}  // namespace icp
