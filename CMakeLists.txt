cmake_minimum_required(VERSION 3.30)
project(neural_network)

set(CMAKE_CXX_STANDARD 20)

include_directories(include)

add_executable(neural_network src/main.cpp
        src/layer.cpp
        include/layer.h
        src/neuron.cpp
        include/neuron.h)