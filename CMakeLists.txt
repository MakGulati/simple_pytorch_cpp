cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(simple_pytorch_cpp)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(simple_pytorch_cpp train.cpp  nn.h csv.h nn.cpp)
target_link_libraries(simple_pytorch_cpp "${TORCH_LIBRARIES}")
set_property(TARGET simple_pytorch_cpp PROPERTY CXX_STANDARD 14)