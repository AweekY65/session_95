// Terminal-driven ICP test suite. All point clouds and ground-truth
// transforms are generated in memory; nothing touches the network or ROS.
#include "icp/icp.hpp"

#include <cmath>
#include <cstdio>
#include <random>
#include <string>

namespace {

int g_failures = 0;
int g_checks = 0;

void Check(bool condition, const std::string& name) {
  ++g_checks;
  if (!condition) {
    ++g_failures;
    std::printf("  [FAIL] %s\n", name.c_str());
  } else {
    std::printf("  [ ok ] %s\n", name.c_str());
  }
}

std::mt19937& Rng() {
  static std::mt19937 gen(42);
  return gen;
}

// Asymmetric Gaussian blobs: distinctive geometry without rotational
// symmetry, so nearest-neighbor correspondences are informative.
icp::PointCloud<3> Scene3(size_t n_per_blob) {
  const Eigen::Vector3d centers[] = {
      {0.0, 0.0, 0.0}, {2.0, 0.5, -0.3}, {-1.2, 1.5, 0.8},
      {0.4, -1.8, 1.2}, {1.0, 1.0, 2.0}, {-0.8, -0.9, -1.5}};
  std::normal_distribution<double> noise(0.0, 0.15);
  icp::PointCloud<3> cloud;
  for (const auto& c : centers) {
    for (size_t i = 0; i < n_per_blob; ++i) {
      cloud.push_back(c + Eigen::Vector3d(noise(Rng()), noise(Rng()), noise(Rng())));
    }
  }
  return cloud;
}

icp::PointCloud<2> Scene2(size_t n_per_blob) {
  const Eigen::Vector2d centers[] = {
      {0.0, 0.0}, {1.8, 0.4}, {-1.1, 1.4}, {0.5, -1.6}, {1.0, 1.2}};
  std::normal_distribution<double> noise(0.0, 0.12);
  icp::PointCloud<2> cloud;
  for (const auto& c : centers) {
    for (size_t i = 0; i < n_per_blob; ++i) {
      cloud.push_back(c + Eigen::Vector2d(noise(Rng()), noise(Rng())));
    }
  }
  return cloud;
}

Eigen::Matrix3d RotationZYX(double roll, double pitch, double yaw) {
  return (Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()) *
          Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) *
          Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX()))
      .toRotationMatrix();
}

template <int Dim>
icp::PointCloud<Dim> Transform(const icp::PointCloud<Dim>& cloud,
                               const Eigen::Matrix<double, Dim, Dim>& r,
                               const Eigen::Matrix<double, Dim, 1>& t) {
  icp::PointCloud<Dim> out(cloud.size());
  for (size_t i = 0; i < cloud.size(); ++i) out[i] = r * cloud[i] + t;
  return out;
}

double RotationError3(const Eigen::Matrix3d& estimated, const Eigen::Matrix3d& truth) {
  const Eigen::Matrix3d diff = estimated * truth.transpose();
  return Eigen::AngleAxisd(diff).angle();
}

template <int Dim>
void CheckProperRotation(const Eigen::Matrix<double, Dim, Dim>& r, const std::string& tag) {
  Check(std::abs(r.determinant() - 1.0) < 1e-9, tag + ": det(R) == +1 (no reflection)");
  Check((r * r.transpose() - Eigen::Matrix<double, Dim, Dim>::Identity()).norm() < 1e-9,
        tag + ": R is orthonormal (no scale)");
}

// Ground-truth rigid transform used by most tests: target = T(source), so
// ICP aligning source onto target must recover T.
struct Case3 {
  Eigen::Matrix3d r;
  Eigen::Vector3d t;
};

Case3 MakeCase3() {
  return {RotationZYX(0.3, -0.2, 0.5), Eigen::Vector3d(0.4, -0.3, 0.2)};
}

void TestNoiseFree3D() {
  std::printf("Test: noise-free 3D alignment\n");
  const auto source = Scene3(50);
  const Case3 gt = MakeCase3();
  const auto target = Transform(source, gt.r, gt.t);

  const auto result = icp::Align<3>(source, target);
  Check(result.status == icp::Status::kConverged, "status converged");
  Check(result.rmse < 1e-8, "rmse ~ 0");
  Check(RotationError3(result.rotation, gt.r) < 1e-6, "rotation error < 1e-6 rad");
  Check((result.translation - gt.t).norm() < 1e-6, "translation error < 1e-6");
  CheckProperRotation<3>(result.rotation, "noise-free");
}

void TestNoisy3D() {
  std::printf("Test: noisy 3D alignment\n");
  const auto source = Scene3(50);
  const Case3 gt = MakeCase3();
  auto target = Transform(source, gt.r, gt.t);
  std::normal_distribution<double> noise(0.0, 0.01);
  for (auto& p : target) p += Eigen::Vector3d(noise(Rng()), noise(Rng()), noise(Rng()));

  const auto result = icp::Align<3>(source, target);
  Check(result.ok(), "status ok");
  Check(RotationError3(result.rotation, gt.r) < 0.02, "rotation error < 0.02 rad");
  Check((result.translation - gt.t).norm() < 0.02, "translation error < 0.02");
  CheckProperRotation<3>(result.rotation, "noisy");
}

void TestOutliers3D() {
  std::printf("Test: outliers rejected by max correspondence distance\n");
  const auto source = Scene3(50);
  const Case3 gt = MakeCase3();
  auto target = Transform(source, gt.r, gt.t);
  // Add far-away outliers to the target that must not pull the estimate.
  std::normal_distribution<double> spread(0.0, 1.0);
  for (int i = 0; i < 60; ++i) {
    target.push_back(Eigen::Vector3d(50.0 + spread(Rng()), 50.0 + spread(Rng()), 50.0));
  }

  icp::Options options;
  options.max_correspondence_distance = 1.0;
  const auto result = icp::Align<3>(source, target, options);
  Check(result.ok(), "status ok");
  Check(RotationError3(result.rotation, gt.r) < 1e-3, "rotation error < 1e-3 rad");
  Check((result.translation - gt.t).norm() < 1e-3, "translation error < 1e-3");
  Check(result.num_correspondences <= static_cast<int>(source.size()),
        "outliers excluded from correspondences");
}

void TestPartialOverlap3D() {
  std::printf("Test: partial overlap\n");
  // Source and target observe different (but overlapping) parts of one scene.
  const auto scene = Scene3(80);
  icp::PointCloud<3> source(scene.begin(), scene.begin() + 340);
  icp::PointCloud<3> target_scene(scene.begin() + 140, scene.end());
  const Case3 gt = MakeCase3();
  const auto target = Transform(target_scene, gt.r, gt.t);

  icp::Options options;
  options.max_correspondence_distance = 1.0;
  const auto result = icp::Align<3>(source, target, options);
  Check(result.ok(), "status ok");
  Check(RotationError3(result.rotation, gt.r) < 0.05, "rotation error < 0.05 rad");
  Check((result.translation - gt.t).norm() < 0.05, "translation error < 0.05");
}

void TestBadInitialGuess3D() {
  std::printf("Test: recovery from a poor initial guess\n");
  const auto source = Scene3(50);
  const Case3 gt = MakeCase3();
  const auto target = Transform(source, gt.r, gt.t);

  // Perturb the true transform by ~25 degrees and a translation offset.
  const Eigen::Matrix3d init_r = gt.r * RotationZYX(0.15, -0.2, 0.25);
  const Eigen::Vector3d init_t = gt.t + Eigen::Vector3d(0.5, 0.4, -0.3);
  const auto result = icp::Align<3>(source, target, icp::Options{}, init_r, init_t);
  Check(result.ok(), "status ok");
  Check(RotationError3(result.rotation, gt.r) < 1e-3, "rotation error < 1e-3 rad");
  Check((result.translation - gt.t).norm() < 1e-3, "translation error < 1e-3");
}

void Test2D() {
  std::printf("Test: 2D alignment\n");
  const auto source = Scene2(40);
  const double angle = 0.6;
  const Eigen::Matrix2d gt_r = Eigen::Rotation2Dd(angle).toRotationMatrix();
  const Eigen::Vector2d gt_t(0.5, -0.4);
  const auto target = Transform(source, gt_r, gt_t);

  const auto result = icp::Align<2>(source, target);
  Check(result.status == icp::Status::kConverged, "status converged");
  Check((result.rotation - gt_r).norm() < 1e-6, "rotation error < 1e-6");
  Check((result.translation - gt_t).norm() < 1e-6, "translation error < 1e-6");
  CheckProperRotation<2>(result.rotation, "2d");
}

void TestEmptyCloud() {
  std::printf("Test: empty point cloud fails cleanly\n");
  const icp::PointCloud<3> empty;
  const auto cloud = Scene3(10);
  const auto r1 = icp::Align<3>(empty, cloud);
  const auto r2 = icp::Align<3>(cloud, empty);
  Check(r1.status == icp::Status::kInvalidInput && !r1.ok(), "empty source rejected");
  Check(r2.status == icp::Status::kInvalidInput && !r2.ok(), "empty target rejected");
}

void TestDegenerateCollinear() {
  std::printf("Test: collinear points reported as degenerate\n");
  icp::PointCloud<3> source(50);
  for (size_t i = 0; i < source.size(); ++i) source[i] = Eigen::Vector3d(i * 0.1, 0, 0);
  const Case3 gt = MakeCase3();
  const auto target = Transform(source, gt.r, gt.t);
  const auto result = icp::Align<3>(source, target);
  Check(result.status == icp::Status::kDegenerateGeometry && !result.ok(),
        "collinear cloud -> degenerate_geometry");
}

void TestDegenerateCoincident() {
  std::printf("Test: coincident points reported as degenerate\n");
  icp::PointCloud<2> source(20, Eigen::Vector2d(1.0, 2.0));
  const auto target = source;
  const auto result = icp::Align<2>(source, target);
  Check(result.status == icp::Status::kDegenerateGeometry && !result.ok(),
        "coincident cloud -> degenerate_geometry");
}

void TestSevereNonOverlap() {
  std::printf("Test: severely non-overlapping clouds fail explicitly\n");
  const auto source = Scene3(20);
  auto target = Scene3(20);
  for (auto& p : target) p += Eigen::Vector3d(100.0, 0.0, 0.0);

  icp::Options options;
  options.max_correspondence_distance = 5.0;
  const auto result = icp::Align<3>(source, target, options);
  Check(result.status == icp::Status::kInsufficientCorrespondences && !result.ok(),
        "far-apart clouds -> insufficient_correspondences");
}

void TestMaxIterationsStatus() {
  std::printf("Test: iteration budget produces explicit status\n");
  const auto source = Scene3(20);
  const Case3 gt = MakeCase3();
  const auto target = Transform(source, gt.r, gt.t);

  icp::Options options;
  options.max_iterations = 1;
  options.tolerance = 0.0;  // impossible to "converge"
  const auto result = icp::Align<3>(source, target, options);
  Check(result.status == icp::Status::kMaxIterationsReached, "status max_iterations_reached");
  Check(result.iterations == 1, "iteration count respected");
}

}  // namespace

int main() {
  TestNoiseFree3D();
  TestNoisy3D();
  TestOutliers3D();
  TestPartialOverlap3D();
  TestBadInitialGuess3D();
  Test2D();
  TestEmptyCloud();
  TestDegenerateCollinear();
  TestDegenerateCoincident();
  TestSevereNonOverlap();
  TestMaxIterationsStatus();

  std::printf("\n%d checks, %d failures\n", g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}
