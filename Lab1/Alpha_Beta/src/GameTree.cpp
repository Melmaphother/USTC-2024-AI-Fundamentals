
#include "GameTree.h"


GameTreeNode
GameTreeNode::CreateChildNode(ChessBoardMatrix &chessboard_matrix, Move move) {
    // 记录父结点到子结点的动作
    Move parent_to_child_move = move;
    // 子结点交换走棋方
    ChessColor child_color = curr_color == Red ? Black : Red;
    // 根据父结点和 move 创建子结点的棋盘
    ChessBoardMatrix child_chessboard_matrix = chessboard_matrix;
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
            if (child_chessboard_matrix[move.next_x][move.next_y] == ChessType::BlackKing) {
                child_is_stop_game = true;
            }
        }
            // 如果当前走棋方是黑子，则吃的是红子
        else {
            if (child_chessboard_matrix[move.next_x][move.next_y] == ChessType::RedKing) {
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
        auto next_chess_type = child_chessboard_matrix[move.next_x][move.next_y];
        auto child_eval_matrix = eval.getChessPowerEvalMatrix(next_chess_type);
        child_curr_score = opponent_score - eval.getChessValue(next_chess_type)
                           - child_eval_matrix[move.next_x][move.next_y];
    } else {
        child_curr_score = opponent_score;
    }

    // 判断是否游戏结束以及计算完分数之后，可以移动棋子
    child_chessboard_matrix[move.next_x][move.next_y] = child_chessboard_matrix[move.init_x][move.init_y];
    child_chessboard_matrix[move.init_x][move.init_y] = ChessType::Empty;

    GameTreeNode child(
            parent_to_child_move,
              child_color, child_chessboard_matrix,
              child_curr_score, child_opponent_score,
              child_is_stop_game,
              child_max_depth, child_curr_depth
              );
    return child;
}

/**
 * node 的分数 = 程序方分数 - 对手方分数
 * 分数大于 0 表示程序方有优势，分数小于 0 表示对手方有优势
 * @return node 的分数
 */
int GameTreeNode::getNodeScore() {
    // 红方为程序方，黑方为对手方
    if (curr_color == ChessColor::Red) {
        return curr_score - opponent_score;
    } else {
        return opponent_score - curr_score;
    }
}

/**
 * 根结点当前走棋方分数的初始化
 * @param _curr_color 当前走棋方
 * @param init_chessboard 初始棋盘
 * @return 当前走棋方初始分数
 */
int GameTreeNode::getInitNodeCurrScore(ChessColor _curr_color, ChessBoardMatrix &init_chessboard_matrix) {
    // 根结点的行棋方为红方，对手方为黑方
    // 此时其分数均为所有棋子的价值 + 所有棋子棋力评估
    Evaluate eval;
    int init_curr_score = eval.getAllChessPowerEval(init_chessboard_matrix, _curr_color) +
                          eval.getAllChessValue(init_chessboard_matrix, _curr_color);
    return init_curr_score;
}


/**
 * 根结点对手方分数的初始化
 * @param _curr_color 根结点对手方
 * @param init_chessboard 初始棋盘
 * @return 对手方初始分数
 */
int GameTreeNode::getInitNodeOpponentScore(ChessColor _curr_color,
                                           ChessBoardMatrix &init_chessboard_matrix) {
    Evaluate eval;
    ChessColor opponent_color = _curr_color == Red ? Black : Red;
    int init_opponent_score = eval.getAllChessPowerEval(init_chessboard_matrix, opponent_color) +
                              eval.getAllChessValue(init_chessboard_matrix, opponent_color);
    return init_opponent_score;
}