cmake_minimum_required(VERSION 3.13)
project(Genetic)

set(CMAKE_CXX_STANDARD 17)

find_package (Python3 COMPONENTS Interpreter Development)
include_directories(${Python_INCLUDE_DIRS})
message(STATUS ${Python_EXECUTABLE})

add_executable(Genetic main.cpp reversi.cpp)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
target_link_libraries(Genetic ${Python_LIBRARIES})