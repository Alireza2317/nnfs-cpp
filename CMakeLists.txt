cmake_minimum_required(VERSION 3.20)
project(NN_From_Scratch VERSION 0.1.0 LANGUAGES CXX)

# For clangd extension
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 REQUIRED)

# activation library (header-only)
add_library(act_lib INTERFACE)
target_include_directories(act_lib INTERFACE include)
target_link_libraries(act_lib INTERFACE Eigen3::Eigen)

# network library (compiled)
add_library(nn_lib src/nn/network.cpp)
target_include_directories(nn_lib PUBLIC include)
target_link_libraries(nn_lib PUBLIC act_lib)



# Create the executable by linking all your files
add_executable(main apps/main.cpp)

target_link_libraries(main PRIVATE nn_lib Eigen3::Eigen)