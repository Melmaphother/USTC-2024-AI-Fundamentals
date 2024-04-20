
#include "Evaluate.h"

Evaluate::Evaluate() {
    red_rook_power_eval_matrix = {
            {-6, 5, -2, 4, 8, 8, 6, 6, 6, 6},
            {6, 8, 8, 9, 12, 11, 13, 8, 12, 8},
            {4, 6, 4, 4, 12, 11, 13, 7, 9, 7},
            {12, 12, 12, 12, 14, 14, 16, 14, 16, 13},
            {0, 0, 12, 14, 15, 15, 16, 16, 33, 14},
            {12, 12, 12, 12, 14, 14, 16, 14, 16, 13},
            {4, 6, 4, 4, 12, 11, 13, 7, 9, 7},
            {6, 8, 8, 9, 12, 11, 13, 8, 12, 8},
            {-6, 5, -2, 4, 8, 8, 6, 6, 6, 6}
    };
    red_knight_power_eval_matrix = {
            {0, -3, 5, 4, 2, 2, 5, 4, 2, 2},
            {-3, 2, 4, 6, 10, 12, 20, 10, 8, 2},
            {2, 4, 6, 10, 13, 11, 12, 11, 15, 2},
            {0, 5, 7, 7, 14, 15, 19, 15, 9, 8},
            {2, -10, 4, 10, 15, 16, 12, 11, 6, 2},
            {0, 5, 7, 7, 14, 15, 19, 15, 9, 8},
            {2, 4, 6, 10, 13, 11, 12, 11, 15, 2},
            {-3, 2, 4, 6, 10, 12, 20, 10, 8, 2},
            {0, -3, 5, 4, 2, 2, 5, 4, 2, 2}
    };
    red_cannon_power_eval_matrix = {
            {0, 0, 1, 0, -1, 0, 0, 1, 2, 4},
            {0, 1, 0, 0, 0, 0, 3, 1, 2, 4},
            {1, 2, 4, 0, 3, 0, 3, 0, 0, 0},
            {3, 2, 3, 0, 0, 0, 2, -5, -4, -5},
            {3, 2, 5, 0, 4, 4, 4, -4, -7, -6},
            {3, 2, 3, 0, 0, 0, 2, -5, -4, -5},
            {1, 2, 4, 0, 3, 0, 3, 0, 0, 0},
            {0, 1, 0, 0, 0, 0, 3, 1, 2, 4},
            {0, 0, 1, 0, -1, 0, 0, 1, 2, 4}
    };
    red_advisor_power_eval_matrix = {
            {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
            {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
            {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
            {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
            {0, 3, 0, 0, 0, 0, 0, 0, 0, 0},
            {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
            {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
            {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
            {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
    };
    red_bishop_power_eval_matrix = {
            {0, 0, -2, 0, 0, 0, 0, 0, 0, 0},
            {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
            {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
            {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
            {0, 0, 3, 0, 0, 0, 0, 0, 0, 0},
            {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
            {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
            {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
            {0, 0, -2, 0, 0, 0, 0, 0, 0, 0}
    };
    red_pawn_power_eval_matrix = {
            {0, 0, 0, -2, 3, 10, 20, 20, 20, 0},
            {0, 0, 0, 0, 0, 18, 27, 30, 30, 0},
            {0, 0, 0, -2, 4, 22, 30, 45, 50, 0},
            {0, 0, 0, 0, 0, 35, 40, 55, 65, 2},
            {0, 0, 0, 6, 7, 40, 42, 55, 70, 4},
            {0, 0, 0, 0, 0, 35, 40, 55, 65, 2},
            {0, 0, 0, -2, 4, 22, 30, 45, 50, 0},
            {0, 0, 0, 0, 0, 18, 27, 30, 30, 0},
            {0, 0, 0, -2, 3, 10, 20, 20, 20, 0}
    };
    red_king_power_eval_matrix = {
            {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
            {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
            {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
            {1, -8, -9, 0, 0, 0, 0, 0, 0, 0},
            {5, -8, -9, 0, 0, 0, 0, 0, 0, 0},
            {1, -8, -9, 0, 0, 0, 0, 0, 0, 0},
            {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
            {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
            {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
    };
    black_rook_power_eval_matrix = getBlackEvalFromRed(RedRook);
    black_knight_power_eval_matrix = getBlackEvalFromRed(RedKnight);
    black_cannon_power_eval_matrix = getBlackEvalFromRed(RedCannon);
    black_advisor_power_eval_matrix = getBlackEvalFromRed(RedAdvisor);
    black_bishop_power_eval_matrix = getBlackEvalFromRed(RedBishop);
    black_pawn_power_eval_matrix = getBlackEvalFromRed(RedPawn);
    black_king_power_eval_matrix = getBlackEvalFromRed(RedKing);

    ChessValue = {
            {RedRook, 500},
            {RedKnight, 300},
            {RedCannon, 300},
            {RedAdvisor, 10},
            {RedBishop, 30},
            {RedPawn, 90},
            {RedKing, 10000},
            {BlackRook, 500},
            {BlackKnight, 300},
            {BlackCannon, 300},
            {BlackAdvisor, 10},
            {BlackBishop, 30},
            {BlackPawn, 90},
            {BlackKing, 10000},
            {Empty, 0}
    };

    MoveValue = {
            {RedRook, 500},
            {RedKnight, 100},
            {RedCannon, 100},
            {RedAdvisor, 0},
            {RedBishop, 0},
            {RedPawn, -20},
            {RedKing, 9999},
            {BlackRook, 500},
            {BlackKnight, 100},
            {BlackCannon, 100},
            {BlackAdvisor, 0},
            {BlackBishop, 0},
            {BlackPawn, -20},
            {BlackKing, 9999},
            {Empty, 0}
    };

}


std::vector<std::vector<int>> Evaluate::getBlackEvalFromRed(ChessType type) {
    std::vector<std::vector<int>> BlackEvaluate;
    auto RedEvaluate = getChessPowerEvalMatrix(type);
    for (int i = 0; i < RedEvaluate.size(); i++) {
        std::vector<int> row;
        row.reserve(RedEvaluate[0].size());
        for (int j = 0; j < RedEvaluate[0].size(); j++) {
            row.push_back(RedEvaluate[RedEvaluate.size() - 1 - i][RedEvaluate[0].size() - 1 - j]);
        }
        BlackEvaluate.push_back(row);
    }
    return BlackEvaluate;
}

/**
 * 棋子所在位置的评估（棋力评估）
 * @param type 棋子类型
 * @return 对应所有位置评估矩阵
 */
std::vector<std::vector<int>> Evaluate::getChessPowerEvalMatrix(ChessType type){
    switch (type) {
        case RedRook:
            return red_rook_power_eval_matrix;
        case RedKnight:
            return red_knight_power_eval_matrix;
        case RedCannon:
            return red_cannon_power_eval_matrix;
        case RedAdvisor:
            return red_advisor_power_eval_matrix;
        case RedBishop:
            return red_bishop_power_eval_matrix;
        case RedPawn:
            return red_pawn_power_eval_matrix;
        case RedKing:
            return red_king_power_eval_matrix;
        case BlackRook:
            return black_rook_power_eval_matrix;
        case BlackKnight:
            return black_knight_power_eval_matrix;
        case BlackCannon:
            return black_cannon_power_eval_matrix;
        case BlackAdvisor:
            return black_advisor_power_eval_matrix;
        case BlackBishop:
            return black_bishop_power_eval_matrix;
        case BlackPawn:
            return black_pawn_power_eval_matrix;
        case BlackKing:
            return black_king_power_eval_matrix;
        case Empty:
            std::cout << "Empty chess type has no evaluate" << std::endl;
            return std::vector<std::vector<int>>(10, std::vector<int>(9, 0));
    }
}

/**
 * 获取棋子价值
 * @param type 棋子类型
 * @return 棋子价值
 */

int Evaluate::getChessValue(ChessType type) {
    if (ChessValue.find(type) == ChessValue.end()) {
        std::cout << "Chess type not found" << std::endl;
        return 0;
    }
    return ChessValue[type];
}

/**
 * 获取行棋价值
 * @param type 吃掉对方的棋子类型
 * @return 吃掉对方棋子可以获得的价值（注意有正有负）
 */
int Evaluate::getMoveValue(ChessType type) {
    if (MoveValue.find(type) == MoveValue.end()) {
        std::cout << "Chess type not found" << std::endl;
        return 0;
    }
    return MoveValue[type];
}


/**
 * 初始化棋盘时的初始评估
 * @param color 当前下棋方
 * @return 当前下棋方的所有棋子的棋力评估之和
 */

int Evaluate::getAllChessPowerEval(ChessBoardMatrix &chessboard_matrix, ChessColor color) {
    int sum = 0;
    for (int i = 0; i < 9; i++) {
        for (int j = 0; j < 10; j++) {
            auto chess_type = chessboard_matrix[i][j];
            if (chess_type == Empty) continue;
            auto chess_color = getChessColor(chess_type);
            if (chess_color == color) {
                auto eval_matrix = getChessPowerEvalMatrix(chess_type);
                sum += eval_matrix[i][j];
            }
        }
    }
    return sum;
}

/**
 * 初始化棋盘时的初始评估
 * @param color 当前下棋方
 * @return 当前下棋方的所有棋子的价值之和
 */

int Evaluate::getAllChessValue(ChessBoardMatrix &chessboard_matrix, ChessColor color) {
    int sum = 0;
    for (int i = 0; i < 9; i++) {
        for (int j = 0; j < 10; j++) {
            auto chess_type = chessboard_matrix[i][j];
            if (chess_type == Empty) continue;
            auto chess_color = getChessColor(chess_type);
            if (chess_color == color) {
                sum += getChessValue(chess_type);
            }
        }
    }
    return sum;
}