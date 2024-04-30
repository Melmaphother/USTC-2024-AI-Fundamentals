#include "Astar.h"
#include <iostream>
#include <string>
#include <chrono>

int main() {
    std::string input_base = "../input/input_";
    std::string output_base = "../output/output_";
    std::string heuristic_function = "non-trivial";
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i <= 10; i++) {
        std::string input_file = input_base + std::to_string(i) + ".txt";
        std::string output_file = output_base + std::to_string(i) + ".txt";
        std::cout << "Processing " << input_file << "..." << std::endl;

        Astar astar(input_file, output_file, heuristic_function);
        astar.AstarSearch();
    }
    auto end = std::chrono::high_resolution_clock::now();

    // 计算持续时间
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // 输出结果
    std::cout << heuristic_function << " time:" << duration.count() << " ms" << std::endl;
}