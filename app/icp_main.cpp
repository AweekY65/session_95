// Local point-cloud ICP rigid registration CLI.
//
// Usage:
//   icp_cli <source.txt> <target.txt> [options]
//
// Point clouds are plain ASCII files with one point per row (2 or 3
// whitespace-separated coordinates). Options:
//   --init <file>    (D+1)x(D+1) homogeneous initial transform (row-major).
//   --out <file>     Write the final transform to this file.
//   --max-dist <d>   Maximum correspondence distance (<=0 disables the gate).
//   --tol <t>        Convergence tolerance on RMSE change.
//   --iters <n>      Maximum number of iterations.
//
// Everything runs locally: no ROS, no PCL, no network access.

#include <Eigen/Dense>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "icp/icp.h"

namespace {

bool LoadMatrix(const std::string& path, Eigen::MatrixXd* matrix) {
  std::ifstream in(path);
  if (!in) {
    std::cerr << "Cannot open file: " << path << "\n";
    return false;
  }
  std::vector<std::vector<double>> rows;
  std::string line;
  size_t cols = 0;
  while (std::getline(in, line)) {
    std::istringstream iss(line);
    std::vector<double> row;
    double value;
    while (iss >> value) row.push_back(value);
    if (row.empty()) continue;
    if (cols == 0) {
      cols = row.size();
    } else if (row.size() != cols) {
      std::cerr << "Inconsistent column count in " << path << "\n";
      return false;
    }
    rows.push_back(std::move(row));
  }
  if (rows.empty()) {
    *matrix = Eigen::MatrixXd(0, static_cast<int>(cols));
    return true;
  }
  matrix->resize(static_cast<int>(rows.size()), static_cast<int>(cols));
  for (size_t i = 0; i < rows.size(); ++i)
    for (size_t j = 0; j < cols; ++j) (*matrix)(i, j) = rows[i][j];
  return true;
}

void PrintUsage(const char* argv0) {
  std::cerr << "Usage: " << argv0 << " <source.txt> <target.txt> "
            << "[--init file] [--out file] [--max-dist d] [--tol t] [--iters n]\n";
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    PrintUsage(argv[0]);
    return EXIT_FAILURE;
  }

  icp::IcpOptions options;
  std::string init_path, out_path;
  const std::string source_path = argv[1];
  const std::string target_path = argv[2];
  for (int i = 3; i < argc; ++i) {
    const std::string flag = argv[i];
    auto next = [&](const char* name) -> std::string {
      if (i + 1 >= argc) {
        std::cerr << "Missing value for " << name << "\n";
        std::exit(EXIT_FAILURE);
      }
      return argv[++i];
    };
    if (flag == "--init") {
      init_path = next("--init");
    } else if (flag == "--out") {
      out_path = next("--out");
    } else if (flag == "--max-dist") {
      options.max_correspondence_distance = std::stod(next("--max-dist"));
    } else if (flag == "--tol") {
      options.tolerance = std::stod(next("--tol"));
    } else if (flag == "--iters") {
      options.max_iterations = std::stoi(next("--iters"));
    } else {
      std::cerr << "Unknown option: " << flag << "\n";
      PrintUsage(argv[0]);
      return EXIT_FAILURE;
    }
  }

  Eigen::MatrixXd source, target, initial;
  if (!LoadMatrix(source_path, &source) || !LoadMatrix(target_path, &target)) {
    return EXIT_FAILURE;
  }
  if (!init_path.empty() && !LoadMatrix(init_path, &initial)) {
    return EXIT_FAILURE;
  }

  const icp::Icp icp(options);
  const icp::IcpResult result = icp.Align(source, target, initial);

  std::cout << "status: " << icp::ToString(result.status) << "\n"
            << "message: " << result.message << "\n"
            << "iterations: " << result.iterations << "\n"
            << "correspondences: " << result.num_correspondences << "\n"
            << "rmse: " << result.rmse << "\n";
  if (result.transform.size() != 0) {
    std::cout << "transform:\n" << result.transform << "\n";
  }

  if (!out_path.empty() && result.transform.size() != 0) {
    std::ofstream out(out_path);
    if (!out) {
      std::cerr << "Cannot write to: " << out_path << "\n";
      return EXIT_FAILURE;
    }
    out << result.transform << "\n";
  }

  return result.Success() ? EXIT_SUCCESS : EXIT_FAILURE;
}
