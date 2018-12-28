cmake_minimum_required(VERSION 3.13)
project(CVersion)

set(CMAKE_CXX_STANDARD 17)

find_package (Python3 COMPONENTS Interpreter Development)
include_directories(${Python_INCLUDE_DIRS})
message(STATUS ${Python_EXECUTABLE})

add_executable(CVersion main.cpp)
target_link_libraries(CVersion ${Python_LIBRARIES})