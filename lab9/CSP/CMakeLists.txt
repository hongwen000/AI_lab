cmake_minimum_required(VERSION 2.8)
set (CMAKE_CXX_STANDARD 14)

project(CSP)
find_package(range-v3)
add_executable(${PROJECT_NAME} "main.cpp")
target_link_libraries(${PROJECT_NAME} range-v3)
