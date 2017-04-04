cmake_minimum_required(VERSION 2.8.8)

project(core)

set(core_src
MultilayerPerceptron.cpp
Network.cpp
)

add_library(${PROJECT_NAME} OBJECT ${core_src})
