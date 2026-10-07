cmake_minimum_required(VERSION 3.16)
project(local_icp CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-Wall -Wextra)

add_library(icp src/icp.cpp)
target_include_directories(icp PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/eigen)

add_executable(icp_cli app/icp_main.cpp)
target_link_libraries(icp_cli PRIVATE icp)

enable_testing()
add_executable(icp_test tests/icp_test.cpp)
target_link_libraries(icp_test PRIVATE icp)
add_test(NAME icp_test COMMAND icp_test)
