cmake_minimum_required(VERSION 2.8.8)

project(demo)

add_executable(DemoMlp DemoMlp.cpp)
target_link_libraries(DemoMlp ${CMAKE_PROJECT_NAME})
