#include "ChessBoard.h"
#include <cassert>
#include <limits>
#include <fstream>

int AlphaBetaSearch(ChessBoard &node, int depth, int alpha, int beta, bool isMaxNode) {
    bool is_my_turn = node.getCurrColor() == Red;
    assert(isMaxNode == is_my_turn);

    if (depth == 0 || node.isStopGame()) {
        return node.getCurrChessBoardScore();
    }

    const std::vector<Move> &moves = node.getAllPossibleMoves();
    if (isMaxNode) {
        for (const Move &move: moves) {
            ChessBoard child_node = node.getChildChessBoardFromMove(move);
            alpha = std::max(alpha, AlphaBetaSearch(child_node, depth - 1, alpha, beta, false));
            if (alpha >= beta) {
                break;
            }
        }
        return alpha;
    } else {
        for (const Move &move: moves) {
            ChessBoard child_node = node.getChildChessBoardFromMove(move);
            beta = std::min(beta, AlphaBetaSearch(child_node, depth - 1, alpha, beta, true));
            if (alpha >= beta) {
                break;
            }
        }
        return beta;
    }
}

Move getBestMoveFromChildren(ChessBoard &node, int depth, int root_score) {
    const std::vector<Move> &moves = node.getAllPossibleMoves();
    for (const Move &move: moves) {
        ChessBoard child_node = node.getChildChessBoardFromMove(move);
        int score = AlphaBetaSearch(child_node, depth - 1, std::numeric_limits<int>::min(),
                                    std::numeric_limits<int>::max(), false);
        if (score == root_score) {
            return move;
        }
    }
    std::cout << "No best move found!" << std::endl;
    return {Empty, -1, -1, -1, -1};
}

void WriteMoveToFile(const std::string &output_file, const Move &move) {
    std::ofstream out(output_file);
    if (!out.is_open()) {
        std::cerr << "Failed to open output file: " << output_file << std::endl;
        return;
    }

    // 如果吃子，输出格式："chess A" at (x1, y1) eats "chess B" at (x2, y2)
    // 如果不吃子，输出格式："chess A" at (x1, y1) moves to (x2, y2)
    if (move.is_eat) {
        out << getChessStringFromType[move.chess_type] << " at (" << move.init_x << ", " << move.init_y << ") eats "
            << getChessStringFromType[move.eat_chess_type] << " at (" << move.next_x << ", " << move.next_y << ")";
    } else {
        out << getChessStringFromType[move.chess_type] << " at (" << move.init_x << ", " << move.init_y << ") moves to ("
            << move.next_x << ", " << move.next_y << ")";
    }

    out.close();
}

int main() {
    std::string input_base = "../input/";
    std::string output_base = "../output/";
    int max_depth = 5;
    for (int i = 1; i <= 10; i++) {
        std::cout << "Case " << i << ": ";
        std::string input_file = input_base + std::to_string(i) + ".txt";
        std::string output_file = output_base + std::to_string(i) + ".txt";

        // 从输入文件中读取棋盘
        ChessBoard chessboard(input_file);
        // AlphaBeta 搜索
        int score = AlphaBetaSearch(chessboard, max_depth, std::numeric_limits<int>::min(),
                                    std::numeric_limits<int>::max(), true);
        std::cout << score << std::endl;
        Move best_move = getBestMoveFromChildren(chessboard, max_depth, score);
        WriteMoveToFile(output_file, best_move);
    }
}
