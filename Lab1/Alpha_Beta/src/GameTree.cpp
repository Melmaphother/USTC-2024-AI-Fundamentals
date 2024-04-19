
#include "GameTree.h"


GameTreeNode GameTreeNode::CreateChildNode(std::vector<std::vector<ChessType>> &chessboard, Move move) {
    // 根据父节点和 move 创建子节点的棋盘
    std::vector<std::vector<ChessType>> child_chessboard = chessboard;
    // 子节点交换走棋方
    ChessColor child_color = curr_color == Red ? Black : Red;
    // 如果没有吃子
    if (move.is_eat) {
        // 如果吃子，这里特殊考虑吃将/帅的情况，这种情况下游戏结束
        // 如果当前走棋方是红子，则吃的是黑子
        if (curr_color == ChessColor::Red) {
            if (child_chessboard[move.next_x][move.next_y] == ChessType::BlackKing) {
                is_stop_game = true;
            }
        }
        // 如果当前走棋方是黑子，则吃的是红子
        else {
            if (child_chessboard[move.next_x][move.next_y] == ChessType::RedKing) {
                is_stop_game = true;
            }
        }
    }
    // 移动棋子
    child_chessboard[move.next_x][move.next_y] = child_chessboard[move.init_x][move.init_y];
    child_chessboard[move.init_x][move.init_y] = ChessType::Empty;

    // 子节点的分数 = 子节点所有棋子的总价值 + move 的分数
    int child_chess_value = Evaluate::getAllChessValueOfColor(child_chessboard, child_color);
    int move_score = move.score;
    int child_score = child_chess_value + move_score;

    // 创建子节点类
    GameTreeNode child(child_color, child_chessboard, child_score, is_stop_game);
    return child;
}

