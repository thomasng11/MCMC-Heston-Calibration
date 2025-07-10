cmake_minimum_required(VERSION 3.14)
project(heston_mcmc)

# Set C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set vcpkg paths
set(CMAKE_TOOLCHAIN_FILE "C:/Users/Thoma/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "Vcpkg toolchain file")
set(CMAKE_PREFIX_PATH "C:/Users/Thoma/vcpkg/installed/x64-windows")

# Set policy for Boost
if(POLICY CMP0167)
  cmake_policy(SET CMP0167 NEW)
endif()

# Find packages
set(Python3_FIND_STRATEGY LOCATION)
set(Python3_FIND_REGISTRY LAST)
find_package(Python3 3.11 EXACT COMPONENTS Interpreter Development REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(Boost REQUIRED COMPONENTS math)

# Add module and link libraries
pybind11_add_module(heston_mcmc src/cpp/heston.cpp src/cpp/mcmc.cpp src/cpp/bindings.cpp)
target_link_libraries(heston_mcmc PRIVATE Eigen3::Eigen Boost::math)

# Set Build Type to Release for optimization
set(CMAKE_BUILD_TYPE Release)