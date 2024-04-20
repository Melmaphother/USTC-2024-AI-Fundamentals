#include "Chess.h"
#include "ChessBoard.h"
#include "AlphaBeta.h"
#include <fstream>

/*
 * Move 结构体为：
 * chess_type: 走棋的棋子
 * init_x: 起始位置 x 坐标
 * init_y: 起始位置 y 坐标
 * next_x: 目标位置 x 坐标
 * next_y: 目标位置 y 坐标
 * score: 走棋的分数（不用输出）
 * is_eat: 是否吃子（判断是否吃子）
 * eat_chess_type: 被吃的棋子（吃子的类型）
 */

// 将最佳路径写入输出文件
void writeBestMovesToFile(const std::string &output_file, std::vector<Move> &best_moves) {
    std::ofstream out(output_file);
    for (auto &move: best_moves) {
        if (move.is_eat) {
            // "chess A" at (x1, y1) eats "chess B" at (x2, y2)
            std::string chess_A = getChessStringFromType[move.chess_type];
            std::string chess_B = getChessStringFromType[move.eat_chess_type];
            out << chess_A << " at (" << move.init_x << ", " << move.init_y << ") eats " << chess_B << " at ("
                << move.next_x << ", " << move.next_y << ")" << std::endl;
        } else {
            // "chess A" at (x1, y1) moves to (x2, y2)
            std::string chess_A = getChessStringFromType[move.chess_type];
            out << chess_A << " at (" << move.init_x << ", " << move.init_y << ") moves to (" << move.next_x << ", "
                << move.next_y << ")" << std::endl;
        }
    }
    out.close();
}

int main() {
    std::string input_base = "../input/";
    std::string output_base = "../output/";
    int max_depth = 20;
    for (int i = 1; i <= 11; i++) {
        std::string input_file = input_base + std::to_string(i) + ".txt";
        std::string output_file = output_base + std::to_string(i) + ".txt";

        // 从输入文件中读取棋盘
        ChessBoard chessboard(input_file);
        ChessBoardMatrix chessboard_matrix = chessboard.getChessBoardMatrix();
        // 构建博弈树和搜索类
        AlphaBeta alpha_beta_search(chessboard_matrix, max_depth);

        // 最终根结点的搜索分数
        int score = alpha_beta_search.AlphaBetaSearch();
        std::cout << "Case " << i << ": " << score << std::endl;
        // 获取最佳路径
        auto best_moves = alpha_beta_search.getBestMoves();

        // 将最佳路径写入输出文件
        writeBestMovesToFile(output_file, best_moves);
    }
}