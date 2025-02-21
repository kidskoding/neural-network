cmake_minimum_required(VERSION 3.30)
project(neural_network)

set(CMAKE_CXX_STANDARD 20)

add_subdirectory(extern/googletest)
include_directories(include)

set(COMMON_SOURCES
        src/neuron.cpp
        src/layer.cpp
)

file(GLOB TESTS "tests/*.cpp")
add_executable(all_tests ${TESTS} ${COMMON_SOURCES})
target_link_libraries(all_tests gtest gtest_main)

add_executable(neural_network 
        src/main.cpp
        ${COMMON_SOURCES}
)