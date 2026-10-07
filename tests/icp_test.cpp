// Terminal-driven automated tests for the local ICP implementation.
// All test data is generated in memory with fixed seeds; no files, network,
// ROS, or PCL are involved.

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "icp/icp.h"

namespace {

int g_failures = 0;
int g_checks = 0;

void Check(bool condition, const std::string& what) {
  ++g_checks;
  if (!condition) {
    ++g_failures;
    std::cout << "  [FAIL] " << what << "\n";
  }
}

#define RUN_TEST(fn)                                    \
  do {                                                  \
    std::cout << "[ RUN  ] " #fn "\n";                  \
    fn();                                               \
    std::cout << "[ DONE ] " #fn "\n";                  \
  } while (0)

Eigen::MatrixXd MakeRandomCloud(int n, int dim, std::mt19937* rng,
                                double extent = 1.0) {
  std::uniform_real_distribution<double> dist(-extent, extent);
  Eigen::MatrixXd cloud(n, dim);
  for (int i = 0; i < n; ++i)
    for (int d = 0; d < dim; ++d) cloud(i, d) = dist(*rng);
  return cloud;
}

Eigen::Matrix3d RotationFromEuler(double roll, double pitch, double yaw) {
  const Eigen::AngleAxisd rx(roll, Eigen::Vector3d::UnitX());
  const Eigen::AngleAxisd ry(pitch, Eigen::Vector3d::UnitY());
  const Eigen::AngleAxisd rz(yaw, Eigen::Vector3d::UnitZ());
  return (rz * ry * rx).toRotationMatrix();
}

Eigen::MatrixXd MakeTransform3d(const Eigen::Matrix3d& r,
                                const Eigen::Vector3d& t) {
  Eigen::MatrixXd tf = Eigen::MatrixXd::Identity(4, 4);
  tf.block<3, 3>(0, 0) = r;
  tf.block<3, 1>(0, 3) = t;
  return tf;
}

Eigen::MatrixXd MakeTransform2d(double angle, const Eigen::Vector2d& t) {
  Eigen::MatrixXd tf = Eigen::MatrixXd::Identity(3, 3);
  tf(0, 0) = std::cos(angle);
  tf(0, 1) = -std::sin(angle);
  tf(1, 0) = std::sin(angle);
  tf(1, 1) = std::cos(angle);
  tf(0, 2) = t.x();
  tf(1, 2) = t.y();
  return tf;
}

Eigen::MatrixXd ApplyTransform(const Eigen::MatrixXd& points,
                               const Eigen::MatrixXd& tf) {
  const int dim = static_cast<int>(points.cols());
  const Eigen::VectorXd t = tf.block(0, dim, dim, 1);
  return ((tf.block(0, 0, dim, dim) * points.transpose()).colwise() + t)
      .transpose();
}

double RotationError(const Eigen::MatrixXd& estimated,
                     const Eigen::MatrixXd& expected, int dim) {
  const Eigen::MatrixXd r_err =
      estimated.block(0, 0, dim, dim).transpose() * expected.block(0, 0, dim, dim);
  // trace(R) = (dim - 2) + 2*cos(angle) for a rotation in 2D or 3D.
  const double cos_angle =
      std::min(1.0, std::max(-1.0, (r_err.trace() - (dim - 2)) / 2.0));
  return std::acos(cos_angle);
}

double TranslationError(const Eigen::MatrixXd& estimated,
                        const Eigen::MatrixXd& expected, int dim) {
  return (estimated.block(0, dim, dim, 1) - expected.block(0, dim, dim, 1)).norm();
}

void CheckRigid(const Eigen::MatrixXd& tf, int dim, const std::string& label) {
  const Eigen::MatrixXd r = tf.block(0, 0, dim, dim);
  Check((r.transpose() * r - Eigen::MatrixXd::Identity(dim, dim)).norm() < 1e-8,
        label + ": rotation is orthonormal (no scaling)");
  Check(std::abs(r.determinant() - 1.0) < 1e-8,
        label + ": det(R) == 1 (no reflection)");
}

void TestExact3d() {
  std::mt19937 rng(42);
  const Eigen::MatrixXd source = MakeRandomCloud(200, 3, &rng);
  const Eigen::MatrixXd gt =
      MakeTransform3d(RotationFromEuler(0.3, -0.2, 0.5), {0.4, -0.3, 0.2});
  const Eigen::MatrixXd target = ApplyTransform(source, gt);

  const icp::IcpResult result = icp::Icp().Align(source, target);
  Check(result.status == icp::IcpStatus::kConverged, "status is kConverged");
  Check(result.rmse >= 0.0 && result.rmse < 1e-8, "RMSE near zero");
  Check(RotationError(result.transform, gt, 3) < 1e-6, "rotation error < 1e-6 rad");
  Check(TranslationError(result.transform, gt, 3) < 1e-6, "translation error < 1e-6");
  Check(result.iterations > 0 && result.iterations <= 50, "iteration count sane");
  CheckRigid(result.transform, 3, "exact-3d");
}

void TestExact2d() {
  std::mt19937 rng(7);
  const Eigen::MatrixXd source = MakeRandomCloud(150, 2, &rng);
  const Eigen::MatrixXd gt = MakeTransform2d(0.4, {0.5, -0.2});
  const Eigen::MatrixXd target = ApplyTransform(source, gt);

  const icp::IcpResult result = icp::Icp().Align(source, target);
  Check(result.status == icp::IcpStatus::kConverged, "status is kConverged");
  Check(result.rmse >= 0.0 && result.rmse < 1e-8, "RMSE near zero");
  Check(RotationError(result.transform, gt, 2) < 1e-6, "rotation error < 1e-6 rad");
  Check(TranslationError(result.transform, gt, 2) < 1e-6, "translation error < 1e-6");
  CheckRigid(result.transform, 2, "exact-2d");
}

void TestNoisy3d() {
  std::mt19937 rng(123);
  const Eigen::MatrixXd source = MakeRandomCloud(300, 3, &rng);
  const Eigen::MatrixXd gt =
      MakeTransform3d(RotationFromEuler(0.2, 0.1, -0.3), {0.3, 0.1, -0.2});
  std::normal_distribution<double> noise(0.0, 0.01);
  Eigen::MatrixXd target = ApplyTransform(source, gt);
  for (int i = 0; i < target.size(); ++i) target(i) += noise(rng);

  const icp::IcpResult result = icp::Icp().Align(source, target);
  Check(result.status == icp::IcpStatus::kConverged, "status is kConverged");
  Check(RotationError(result.transform, gt, 3) < 0.05, "rotation error < 0.05 rad");
  Check(TranslationError(result.transform, gt, 3) < 0.05, "translation error < 0.05");
  Check(result.rmse > 0.0 && result.rmse < 0.05, "RMSE reflects noise level");
  CheckRigid(result.transform, 3, "noisy-3d");
}

void TestOutliers3d() {
  std::mt19937 rng(99);
  Eigen::MatrixXd source = MakeRandomCloud(200, 3, &rng);
  const Eigen::MatrixXd gt =
      MakeTransform3d(RotationFromEuler(0.15, -0.1, 0.2), {0.2, 0.1, 0.1});
  const Eigen::MatrixXd target = ApplyTransform(source, gt);
  // Append gross outliers to the source cloud (no counterpart in target).
  std::uniform_real_distribution<double> far(5.0, 10.0);
  Eigen::MatrixXd corrupted(240, 3);
  corrupted.topRows(200) = source;
  for (int i = 200; i < 240; ++i)
    for (int d = 0; d < 3; ++d) corrupted(i, d) = far(rng);

  icp::IcpOptions options;
  options.max_correspondence_distance = 0.5;
  const icp::IcpResult result = icp::Icp(options).Align(corrupted, target);
  Check(result.status == icp::IcpStatus::kConverged, "status is kConverged");
  Check(result.num_correspondences <= 200, "outliers were rejected by the gate");
  Check(RotationError(result.transform, gt, 3) < 1e-4, "rotation error < 1e-4 rad");
  Check(TranslationError(result.transform, gt, 3) < 1e-4,
        "translation error < 1e-4");
  CheckRigid(result.transform, 3, "outliers-3d");
}

void TestPartialOverlap3d() {
  std::mt19937 rng(2024);
  const Eigen::MatrixXd source = MakeRandomCloud(300, 3, &rng);
  const Eigen::MatrixXd gt =
      MakeTransform3d(RotationFromEuler(0.1, 0.15, -0.1), {0.15, -0.1, 0.05});
  const Eigen::MatrixXd transformed = ApplyTransform(source, gt);
  // Target keeps only 70% of the transformed points and adds target-only
  // points; source keeps 30% non-overlapping points as well.
  Eigen::MatrixXd target(210 + 40, 3);
  target.topRows(210) = transformed.topRows(210);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  for (int i = 210; i < target.rows(); ++i)
    for (int d = 0; d < 3; ++d) target(i, d) = dist(rng);

  icp::IcpOptions options;
  options.max_correspondence_distance = 0.3;
  const icp::IcpResult result = icp::Icp(options).Align(source, target);
  Check(result.Success(), "status indicates success");
  Check(RotationError(result.transform, gt, 3) < 0.02, "rotation error < 0.02 rad");
  Check(TranslationError(result.transform, gt, 3) < 0.02,
        "translation error < 0.02");
  CheckRigid(result.transform, 3, "partial-overlap-3d");
}

void TestPoorInitialGuess3d() {
  std::mt19937 rng(555);
  const Eigen::MatrixXd source = MakeRandomCloud(300, 3, &rng);
  const Eigen::MatrixXd gt =
      MakeTransform3d(RotationFromEuler(0.4, -0.3, 0.35), {0.5, 0.3, -0.4});
  const Eigen::MatrixXd target = ApplyTransform(source, gt);
  // Initial guess is noticeably off from identity (and from the truth).
  const Eigen::MatrixXd init =
      MakeTransform3d(RotationFromEuler(0.15, 0.1, -0.1), {0.2, -0.1, 0.15});

  icp::IcpOptions options;
  options.max_iterations = 100;
  const icp::IcpResult result = icp::Icp(options).Align(source, target, init);
  Check(result.Success(), "status indicates success");
  Check(RotationError(result.transform, gt, 3) < 1e-4, "rotation error < 1e-4 rad");
  Check(TranslationError(result.transform, gt, 3) < 1e-4,
        "translation error < 1e-4");
  CheckRigid(result.transform, 3, "poor-initial-3d");
}

void TestEmptyInput() {
  const Eigen::MatrixXd empty(0, 3);
  const Eigen::MatrixXd cloud = Eigen::MatrixXd::Random(10, 3);

  const icp::IcpResult r1 = icp::Icp().Align(empty, cloud);
  Check(r1.status == icp::IcpStatus::kEmptyInput, "empty source -> kEmptyInput");
  Check(!r1.Success(), "empty source is a failure");
  Check(r1.rmse == -1.0, "empty source reports rmse -1");

  const icp::IcpResult r2 = icp::Icp().Align(cloud, empty);
  Check(r2.status == icp::IcpStatus::kEmptyInput, "empty target -> kEmptyInput");
}

void TestDimensionMismatch() {
  const Eigen::MatrixXd cloud2d = Eigen::MatrixXd::Random(10, 2);
  const Eigen::MatrixXd cloud3d = Eigen::MatrixXd::Random(10, 3);
  const Eigen::MatrixXd cloud4d = Eigen::MatrixXd::Random(10, 4);

  const icp::IcpResult r1 = icp::Icp().Align(cloud2d, cloud3d);
  Check(r1.status == icp::IcpStatus::kDimensionMismatch,
        "2d vs 3d -> kDimensionMismatch");
  const icp::IcpResult r2 = icp::Icp().Align(cloud4d, cloud4d);
  Check(r2.status == icp::IcpStatus::kDimensionMismatch,
        "4d -> kDimensionMismatch");
}

void TestDegenerateCollinear() {
  // All points lie on a single line: rotation about the line is unobservable.
  Eigen::MatrixXd source(50, 3);
  for (int i = 0; i < 50; ++i) source.row(i) = Eigen::RowVector3d(0.1 * i, 0, 0);
  const Eigen::MatrixXd gt =
      MakeTransform3d(RotationFromEuler(0.2, 0.1, 0.3), {0.3, 0.2, 0.1});
  const Eigen::MatrixXd target = ApplyTransform(source, gt);

  const icp::IcpResult result = icp::Icp().Align(source, target);
  Check(result.status == icp::IcpStatus::kDegenerateInput,
        "collinear input -> kDegenerateInput");
  Check(!result.Success(), "degenerate input is a failure");

  // Coincident points are degenerate too.
  const Eigen::MatrixXd points = Eigen::MatrixXd::Zero(20, 3);
  const icp::IcpResult r2 = icp::Icp().Align(points, points);
  Check(r2.status == icp::IcpStatus::kDegenerateInput,
        "coincident input -> kDegenerateInput");
}

void TestSevereNonOverlap() {
  std::mt19937 rng(8);
  const Eigen::MatrixXd source = MakeRandomCloud(100, 3, &rng);
  // Target lives in a completely different region of space.
  const Eigen::MatrixXd target = MakeRandomCloud(100, 3, &rng).rowwise() +
                                 Eigen::RowVector3d(100.0, 100.0, 100.0);

  icp::IcpOptions options;
  options.max_correspondence_distance = 1.0;
  const icp::IcpResult result = icp::Icp(options).Align(source, target);
  Check(result.status == icp::IcpStatus::kInsufficientCorrespondences,
        "non-overlapping clouds -> kInsufficientCorrespondences");
  Check(!result.Success(), "severe non-overlap is a failure");
  Check(result.rmse == -1.0, "no valid correspondence set -> rmse -1");
}

}  // namespace

int main() {
  RUN_TEST(TestExact3d);
  RUN_TEST(TestExact2d);
  RUN_TEST(TestNoisy3d);
  RUN_TEST(TestOutliers3d);
  RUN_TEST(TestPartialOverlap3d);
  RUN_TEST(TestPoorInitialGuess3d);
  RUN_TEST(TestEmptyInput);
  RUN_TEST(TestDimensionMismatch);
  RUN_TEST(TestDegenerateCollinear);
  RUN_TEST(TestSevereNonOverlap);

  std::cout << "\n" << (g_checks - g_failures) << "/" << g_checks
            << " checks passed\n";
  if (g_failures > 0) {
    std::cout << "RESULT: FAIL (" << g_failures << " failed checks)\n";
    return EXIT_FAILURE;
  }
  std::cout << "RESULT: PASS\n";
  return EXIT_SUCCESS;
}
