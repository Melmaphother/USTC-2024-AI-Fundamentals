#include "AlphaBeta.h"
#include <limits>
#include <chrono>
#include <algorithm>

std::pair<bool, std::string> getCmdOption(char **begin, char **end,
                                          const std::string &option) {
    char **itr = std::find(begin, end, option);
    if (itr != end && ++itr != end) { return std::make_pair(true, *itr); }
    return std::make_pair(false, "");
}

bool cmdOptionExists(char **begin, char **end, const std::string &option) {
    return std::find(begin, end, option) != end;
}

int main(int argc, char **argv) {
    auto help_info = cmdOptionExists(argv, argv + argc, "-h")
            || cmdOptionExists(argv, argv + argc, "--help");
    if (help_info) {
        std::cout << "-h, --help: Show help information" << std::endl;
        std::cout << "--depth: Specify the depth of the search tree" << std::endl;
        std::cout << "-m, --multi-thread: Use multi-thread to speed up the search" << std::endl;
        std::cout << "--input: Specify the base input file path" << std::endl;
        std::cout << "--output: Specify the base output file path" << std::endl;
    }

    auto is_multi_thread = cmdOptionExists(argv, argv + argc, "--multi-thread")
                           || cmdOptionExists(argv, argv + argc, "-m");

    int max_depth;
    auto get_depth = getCmdOption(argv, argv + argc, "--depth");
    if (!get_depth.first) {
        std::cerr << "Please specify the depth!" << std::endl;
        return 0;
    } else {
        max_depth = std::stoi(get_depth.second);
        // std::cout << "Depth: " << get_depth.second << std::endl;
    }

    std::string input_base = "../input/";
    std::string output_base = "../output/output_";
    auto get_input = getCmdOption(argv, argv + argc, "--input");
    auto get_output = getCmdOption(argv, argv + argc, "--output");
    if (get_input.first) {
        input_base = get_input.second;
    }
    if (get_output.first) {
        output_base = get_output.second;
    }

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