cmake_minimum_required(VERSION 3.28)
project(Astar)

set(CMAKE_CXX_STANDARD 14)

include_directories(src)

add_executable(Astar
        src/Astar.cpp
        src/Map.cpp
        src/Map.h
        src/Point.h
        src/Astar.h
        src/main.cpp)
