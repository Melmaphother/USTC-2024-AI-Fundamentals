#include "AlphaBeta.h"
#include <limits>
#include <chrono>

int main() {
    std::string input_base = "../input/";
    std::string output_base = "../output/";
    int max_depth = 5;
    bool is_multi_thread = true;

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 1; i <= 10; i++) {
        std::cout << "Case " << i << ": ";
        std::string input_file = input_base + std::to_string(i) + ".txt";
        std::string output_file = output_base + std::to_string(i) + ".txt";

        // 从输入文件中读取棋盘
        ChessBoard chessboard(input_file);
        // AlphaBeta 搜索
        if (is_multi_thread) {
            auto best_result = AlphaBetaMultiThreadSearch(chessboard, max_depth, std::numeric_limits<int>::min(),
                                                          std::numeric_limits<int>::max(), true);
            std::cout << best_result.first << std::endl;
            WriteMoveToFile(output_file, best_result.second);
        } else {
            int best_score = AlphaBetaSearch(chessboard, max_depth, std::numeric_limits<int>::min(),
                                             std::numeric_limits<int>::max(), true);
            std::cout << best_score << std::endl;
            Move best_move = getBestMoveFromChildren(chessboard, max_depth, best_score);
            WriteMoveToFile(output_file, best_move);
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "Total time: " << std::chrono::duration_cast<std::chrono::seconds>(t1 - t0).count() << "s"
              << std::endl;
    // depth = 5, multi-thread: 91s
    // depth = 5, no multi-thread: 219s
    // depth = 4, multi-thread: 6s
    // depth = 4, no multi-thread: 17s
}