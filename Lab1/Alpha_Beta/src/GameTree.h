
#ifndef ALPHA_BETA_GAME_TREE_H
#define ALPHA_BETA_GAME_TREE_H

#include "Chess.h"
#include "ChessBoard.h"

class GameTreeNode {
private:
    ChessColor curr_color;  // 当前走棋方
    ChessBoard curr_chessboard;  // 当前棋盘
    std::vector<GameTreeNode> children;  // 子结点
    int curr_score{0};  // 当前结点的分数
    int opponent_score{0};  // 对手结点的分数
    bool curr_is_stop_game{false};  // 是否结束游戏

    int max_depth{10};  // 最大搜索深度
    int curr_depth{0};  // 当前搜索深度

public:
    GameTreeNode(ChessColor color, std::vector<std::vector<ChessType>> init_chessboard, int curr_score,
                 int opponent_score, bool is_stop_game, int max_depth, int curr_depth) :
            curr_color(color), curr_chessboard(init_chessboard, color), curr_score(curr_score),
            opponent_score(opponent_score), curr_is_stop_game(is_stop_game) {
        // 如果游戏还没有结束，通过当前棋盘获取所有可以走的动作
        auto moves = curr_chessboard.getMoves();
        if (moves.empty()) {
            // 困毙的特殊情况，游戏结束
            curr_is_stop_game = true;
        }
        // 如果游戏没有结束，且当前搜索深度小于最大搜索深度，创建子结点
        if (!curr_is_stop_game && curr_depth < max_depth) {
            auto chessboard = curr_chessboard.getChessBoard();
            for (auto &move: moves) {
                GameTreeNode child = CreateChildNode(chessboard, move);
                children.push_back(child);
            }
        }
    }

    // 根据当前棋盘、parent 的当前分数和 parent 的对手分数，创建子结点
    GameTreeNode
    CreateChildNode(std::vector<std::vector<ChessType>> &chessboard, Move move);

    // 根据当前颜色，当前行棋方分数和对手分数，计算当前结点的估值
    int getNodeScore();

    ~GameTreeNode() = default;
};


#endif //ALPHA_BETA_GAME_TREE_H
