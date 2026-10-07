cmake_minimum_required(VERSION 3.16)
project(local_icp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(icp src/icp.cpp)
target_include_directories(icp PUBLIC include third_party/eigen-3.4.0)

add_executable(icp_tests tests/test_icp.cpp)
target_link_libraries(icp_tests PRIVATE icp)

enable_testing()
add_test(NAME icp_tests COMMAND icp_tests)
