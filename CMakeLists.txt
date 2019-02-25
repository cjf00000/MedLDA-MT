cmake_minimum_required(VERSION 3.2)
project(MedLDA)

add_subdirectory(gflags)

set (CMAKE_CXX_FLAGS "-march=native -O3 -std=c++1y -Wall -g ${CMAKE_CXX_FLAGS}")


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++1y")

add_subdirectory(MedLDA)
