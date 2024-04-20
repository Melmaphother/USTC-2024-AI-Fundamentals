
#include "GameTree.h"


GameTreeNode
GameTreeNode::CreateChildNode(std::vector<std::vector<ChessType>> &chessboard, Move move) {
    // 子结点交换走棋方
    ChessColor child_color = curr_color == Red ? Black : Red;
    // 根据父结点和 move 创建子结点的棋盘
    std::vector<std::vector<ChessType>> child_chessboard = chessboard;
    // 子结点是否结束游戏
    bool child_is_stop_game = false;
    // 子结点当前分数
    int child_curr_score;
    // 子结点对手分数
    int child_opponent_score;
    // 子结点最大搜索深度，维持父结点的最大搜索深度
    int child_max_depth = max_depth;
    // 子结点当前搜索深度，为父结点的当前搜索深度加 1
    int child_curr_depth = curr_depth + 1;

    // 评估当前棋盘
    Evaluate eval;

    // 判断是否结束游戏
    if (move.is_eat) {
        // 如果吃子，这里特殊考虑吃将/帅的情况，这种情况下游戏结束
        // 如果当前走棋方是红子，则吃的是黑子
        if (curr_color == ChessColor::Red) {
            if (child_chessboard[move.next_x][move.next_y] == ChessType::BlackKing) {
                child_is_stop_game = true;
            }
        }
            // 如果当前走棋方是黑子，则吃的是红子
        else {
            if (child_chessboard[move.next_x][move.next_y] == ChessType::RedKing) {
                child_is_stop_game = true;
            }
        }
    }

    // 子结点当前行棋方是父结点的对手
    // 子结点的对手方是父结点的当前行棋方，显然，父结点的当前行棋方在行棋时，棋子不会减少
    // 子结点对手方分数 = 父结点行棋方当前分数 + move 的影响（包括吃子的加分和位置移动的分数变化）
    child_opponent_score = curr_score + move.score;

    // 子结点当前行棋方分数分两种情况：
    // 如果 move 没有吃子，那么子结点当前行棋方分数 = 父结点对手方分数
    // 如果被吃子，那么子结点当前行棋方分数 = 父结点对手方分数 - 被吃子的价值 - 被吃子原位的棋力评估
    if (move.is_eat) {
        auto child_eval_matrix = eval.getChessPowerEvalMatrix(child_chessboard[move.next_x][move.next_y]);
        child_curr_score = opponent_score - eval.getChessValue(child_chessboard[move.next_x][move.next_y])
                           - child_eval_matrix[move.next_x][move.next_y];
    } else {
        child_curr_score = opponent_score;
    }

    // 判断是否游戏结束以及计算完分数之后，可以移动棋子
    child_chessboard[move.next_x][move.next_y] = child_chessboard[move.init_x][move.init_y];
    child_chessboard[move.init_x][move.init_y] = ChessType::Empty;

    GameTreeNode child(child_color, child_chessboard, child_curr_score, child_opponent_score, child_is_stop_game,
                       child_max_depth, child_curr_depth);
    return child;
}

int GameTreeNode::getNodeScore() {
    // node 的分数 = 程序方分数 - 对手方分数
    // 红方为程序方，黑方为对手方
    if (curr_color == ChessColor::Red) {
        return curr_score - opponent_score;
    } else {
        return opponent_score - curr_score;
    }
}