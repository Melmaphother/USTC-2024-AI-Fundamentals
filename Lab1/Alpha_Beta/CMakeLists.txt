cmake_minimum_required(VERSION 3.28)
project(Alpha_Beta)

set(CMAKE_CXX_STANDARD 14)

include_directories(src)

add_executable(Alpha_Beta
        src/AlphaBeta.cpp
        src/AlphaBeta.h
        src/Chess.h
        src/ChessBoard.cpp
        src/ChessBoard.h
        src/Evaluate.cpp
        src/Evaluate.h
        src/main.cpp
)