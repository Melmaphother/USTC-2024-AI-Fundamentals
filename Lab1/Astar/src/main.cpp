#include "Astar.h"
#include <iostream>
#include <string>

int main() {
    std::string input_base = "../input/input_";
    std::string output_base = "../output/output_";
    for (int i = 0; i <= 10; i++) {
        std::string input_file = input_base + std::to_string(i) + ".txt";
        std::string output_file = output_base + std::to_string(i) + ".txt";
        std::cout << "Processing " << input_file << "..." << std::endl;
        Astar astar(input_file, output_file, "non-trivial");
        astar.AstarSearch();
    }
}