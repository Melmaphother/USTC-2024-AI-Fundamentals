#include "AlphaBeta.h"
#include <fstream>
#include <thread>

int alphaBetaSearch(ChessBoard &node, int depth, int alpha, int beta, bool isMaxNode) {
    bool is_my_turn = node.getCurrColor() == Red;
    if (isMaxNode != is_my_turn) {
        throw std::runtime_error("isMaxNode is not consistent with current color!");
    }

    if (depth == 0 || node.isStopGame()) {
        return node.getCurrChessBoardScore();
    }

    const std::vector<Move> &moves = node.getAllPossibleMoves();
    if (isMaxNode) {
        for (const Move &move: moves) {
            ChessBoard child_node = node.getChildChessBoardFromMove(move);
            alpha = std::max(alpha, alphaBetaSearch(child_node, depth - 1, alpha, beta, false));
            if (alpha >= beta) {
                break;
            }
        }
        return alpha;
    } else {
        for (const Move &move: moves) {
            ChessBoard child_node = node.getChildChessBoardFromMove(move);
            beta = std::min(beta, alphaBetaSearch(child_node, depth - 1, alpha, beta, true));
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
        int score = alphaBetaSearch(child_node, depth - 1, std::numeric_limits<int>::min(),
                                    std::numeric_limits<int>::max(), false);
        if (score == root_score) {
            return move;
        }
    }
    std::cout << "No best move found!" << std::endl;
    return {Empty, -1, -1, -1, -1};
}

std::pair<int, Move> alphaBetaSearchMultiThreads(ChessBoard &node, int depth, int alpha, int beta, bool isMaxNode) {
    bool is_my_turn = node.getCurrColor() == Red;
    if (isMaxNode != is_my_turn) {
        throw std::runtime_error("isMaxNode is not consistent with current color!");
    }
    if (depth < 1) {
        throw std::runtime_error("Depth should be at least 1!");
    }

    if (node.isStopGame()) {
        return {END_GAME_SCORE, {Empty, -1, -1, -1, -1}};
    }

    const std::vector<Move> &moves = node.getAllPossibleMoves();
    std::vector<std::thread> threads(moves.size());
    std::vector<int> scores(moves.size());
    for (int i = 0; i < moves.size(); i++) {
        threads.emplace_back([&, i]() {
            ChessBoard child_node = node.getChildChessBoardFromMove(moves[i]);
            scores[i] = alphaBetaSearch(child_node, depth - 1, alpha, beta, !isMaxNode);
        });
    }
    // 等待所有线程结束
    for (auto &thread: threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }

    int best_score = isMaxNode ? std::numeric_limits<int>::min() : std::numeric_limits<int>::max();
    int best_move_index = -1;
    for (int i = 0; i < moves.size(); i++) {
        if ((isMaxNode && scores[i] > best_score) || (!isMaxNode && scores[i] < best_score)) {
            best_score = scores[i];
            best_move_index = i;
        }
    }
    return {best_score, moves[best_move_index]};
}

void writeMoveToFile(const std::string &output_file, const Move &move) {
    std::ofstream out(output_file);
    if (!out.is_open()) {
        std::cerr << "Failed to open output file: " << output_file << std::endl;
        return;
    }

    // 如果吃子，输出格式："chess A" at (x1, y1) eats "chess B" at (x2, y2)
    // 如果不吃子，输出格式："chess A" at (x1, y1) moves to (x2, y2)
//    if (move.is_eat) {
//        out << getChessNameFromType[move.chess_type] << " at (" << move.init_x << ", " << move.init_y << ") eats "
//            << getChessNameFromType[move.eat_chess_type] << " at (" << move.next_x << ", " << move.next_y << ")";
//    } else {
//        out << getChessNameFromType[move.chess_type] << " at (" << move.init_x << ", " << move.init_y << ") moves to ("
//            << move.next_x << ", " << move.next_y << ")";
//    }
    // 直接输出为 棋子 (x1, y1) (x2, y2)
    out << getCharFromChessType[move.chess_type] << " (" << move.init_x << ", " << move.init_y << ") (" << move.next_x
        << ", " << move.next_y << ")";

    out.close();
}

