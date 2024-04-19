
#ifndef ALPHA_BETA_GAMETREE_H
#define ALPHA_BETA_GAMETREE_H

#include "Chess.h"
#include "ChessBoard.h"

class GameTreeNode {
private:
    ChessColor curr_color;  // 当前走棋方
    ChessBoard curr_chessboard;  // 当前棋盘
    std::vector<GameTreeNode> childrens;  // 子节点
    int curr_score{0};  // 当前节点的分数
    bool is_stop_game{false};  // 是否结束游戏

public:
    GameTreeNode(ChessColor color, std::vector<std::vector<ChessType>> init_chessboard, int score, bool is_stop_game) :
            curr_color(color), curr_chessboard(init_chessboard, color), curr_score(score), is_stop_game(is_stop_game) {
        // 如果游戏还没有结束，通过当前棋盘获取所有可以走的动作
        if (!is_stop_game) {
            auto moves = curr_chessboard.getMoves();
            auto chessboard = curr_chessboard.getChessBoard();
            // 困毙的特殊情况
            if (moves.empty()) {
                is_stop_game = true;
            }

            for (auto &move: moves) {
                GameTreeNode child = CreateChildNode(chessboard, move);
                childrens.push_back(child);
            }
        }
    }

    // 创建子节点
    GameTreeNode CreateChildNode(std::vector<std::vector<ChessType>>& chessboard, Move move);

    ~GameTreeNode() = default;
};


#endif //ALPHA_BETA_GAMETREE_H
