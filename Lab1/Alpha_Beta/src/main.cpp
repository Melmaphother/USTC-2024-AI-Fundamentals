#include "AlphaBeta.h"
#include <limits>
#include <chrono>

int main() {
    std::string input_base = "../input/";
    std::string output_base = "../output/";
    int max_depth = 5;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 1; i <= 10; i++) {
        std::cout << "Case " << i << ": ";
        std::string input_file = input_base + std::to_string(i) + ".txt";
        std::string output_file = output_base + std::to_string(i) + ".txt";

        // 从输入文件中读取棋盘
        ChessBoard chessboard(input_file);
        // AlphaBeta 搜索
        auto best_result = AlphaBetaMultiThreadSearch(chessboard, max_depth, std::numeric_limits<int>::min(),
                                                      std::numeric_limits<int>::max(), true);
        std::cout << best_result.first << std::endl;
        WriteMoveToFile(output_file, best_result.second);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "Total time: " << std::chrono::duration_cast<std::chrono::seconds>(t1 - t0).count() << "s"
              << std::endl;
}