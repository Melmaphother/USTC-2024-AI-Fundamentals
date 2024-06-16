#include "Evaluate.h"

EvalMatrix red_rook_power_eval_matrix = {
        {-6, 5,  -2, 4,  8,  8,  6,  6,  6,  6},
        {6,  8,  8,  9,  12, 11, 13, 8,  12, 8},
        {4,  6,  4,  4,  12, 11, 13, 7,  9,  7},
        {12, 12, 12, 12, 14, 14, 16, 14, 16, 13},
        {0,  0,  12, 14, 15, 15, 16, 16, 33, 14},
        {12, 12, 12, 12, 14, 14, 16, 14, 16, 13},
        {4,  6,  4,  4,  12, 11, 13, 7,  9,  7},
        {6,  8,  8,  9,  12, 11, 13, 8,  12, 8},
        {-6, 5,  -2, 4,  8,  8,  6,  6,  6,  6}
};
EvalMatrix red_knight_power_eval_matrix = {
        {0,  -3,  5, 4,  2,  2,  5,  4,  2,  2},
        {-3, 2,   4, 6,  10, 12, 20, 10, 8,  2},
        {2,  4,   6, 10, 13, 11, 12, 11, 15, 2},
        {0,  5,   7, 7,  14, 15, 19, 15, 9,  8},
        {2,  -10, 4, 10, 15, 16, 12, 11, 6,  2},
        {0,  5,   7, 7,  14, 15, 19, 15, 9,  8},
        {2,  4,   6, 10, 13, 11, 12, 11, 15, 2},
        {-3, 2,   4, 6,  10, 12, 20, 10, 8,  2},
        {0,  -3,  5, 4,  2,  2,  5,  4,  2,  2}
};
EvalMatrix red_cannon_power_eval_matrix = {
        {0, 0, 1, 0, -1, 0, 0, 1,  2,  4},
        {0, 1, 0, 0, 0,  0, 3, 1,  2,  4},
        {1, 2, 4, 0, 3,  0, 3, 0,  0,  0},
        {3, 2, 3, 0, 0,  0, 2, -5, -4, -5},
        {3, 2, 5, 0, 4,  4, 4, -4, -7, -6},
        {3, 2, 3, 0, 0,  0, 2, -5, -4, -5},
        {1, 2, 4, 0, 3,  0, 3, 0,  0,  0},
        {0, 1, 0, 0, 0,  0, 3, 1,  2,  4},
        {0, 0, 1, 0, -1, 0, 0, 1,  2,  4}
};
EvalMatrix red_advisor_power_eval_matrix = {
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
EvalMatrix red_bishop_power_eval_matrix = {
        {0, 0, -2, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0,  0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0,  0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0,  0, 0, 0, 0, 0, 0, 0},
        {0, 0, 3,  0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0,  0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0,  0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0,  0, 0, 0, 0, 0, 0, 0},
        {0, 0, -2, 0, 0, 0, 0, 0, 0, 0}
};
EvalMatrix red_pawn_power_eval_matrix = {
        {0, 0, 0, -2, 3, 10, 20, 20, 20, 0},
        {0, 0, 0, 0,  0, 18, 27, 30, 30, 0},
        {0, 0, 0, -2, 4, 22, 30, 45, 50, 0},
        {0, 0, 0, 0,  0, 35, 40, 55, 65, 2},
        {0, 0, 0, 6,  7, 40, 42, 55, 70, 4},
        {0, 0, 0, 0,  0, 35, 40, 55, 65, 2},
        {0, 0, 0, -2, 4, 22, 30, 45, 50, 0},
        {0, 0, 0, 0,  0, 18, 27, 30, 30, 0},
        {0, 0, 0, -2, 3, 10, 20, 20, 20, 0}
};
EvalMatrix red_king_power_eval_matrix = {
        {0, 0,  0,  0, 0, 0, 0, 0, 0, 0},
        {0, 0,  0,  0, 0, 0, 0, 0, 0, 0},
        {0, 0,  0,  0, 0, 0, 0, 0, 0, 0},
        {1, -8, -9, 0, 0, 0, 0, 0, 0, 0},
        {5, -8, -9, 0, 0, 0, 0, 0, 0, 0},
        {1, -8, -9, 0, 0, 0, 0, 0, 0, 0},
        {0, 0,  0,  0, 0, 0, 0, 0, 0, 0},
        {0, 0,  0,  0, 0, 0, 0, 0, 0, 0},
        {0, 0,  0,  0, 0, 0, 0, 0, 0, 0}
};
EvalMatrix black_rook_power_eval_matrix = {
        {6,  6,  6,  6,  8,  8,  4,  -2, 5,  -6,},
        {8,  12, 8,  13, 11, 12, 9,  8,  8,  6,},
        {7,  9,  7,  13, 11, 12, 4,  4,  6,  4,},
        {13, 16, 14, 16, 14, 14, 12, 12, 12, 12,},
        {14, 33, 16, 16, 15, 15, 14, 12, 0,  0,},
        {13, 16, 14, 16, 14, 14, 12, 12, 12, 12,},
        {7,  9,  7,  13, 11, 12, 4,  4,  6,  4,},
        {8,  12, 8,  13, 11, 12, 9,  8,  8,  6,},
        {6,  6,  6,  6,  8,  8,  4,  -2, 5,  -6,}
};
EvalMatrix black_knight_power_eval_matrix = {
        {2, 2,  4,  5,  2,  2,  4,  5, -3,  0,},
        {2, 8,  10, 20, 12, 10, 6,  4, 2,   -3,},
        {2, 15, 11, 12, 11, 13, 10, 6, 4,   2,},
        {8, 9,  15, 19, 15, 14, 7,  7, 5,   0,},
        {2, 6,  11, 12, 16, 15, 10, 4, -10, 2,},
        {8, 9,  15, 19, 15, 14, 7,  7, 5,   0,},
        {2, 15, 11, 12, 11, 13, 10, 6, 4,   2,},
        {2, 8,  10, 20, 12, 10, 6,  4, 2,   -3,},
        {2, 2,  4,  5,  2,  2,  4,  5, -3,  0,}
};
EvalMatrix black_cannon_power_eval_matrix = {
        {4,  2,  1,  0, 0, -1, 0, 1, 0, 0,},
        {4,  2,  1,  3, 0, 0,  0, 0, 1, 0,},
        {0,  0,  0,  3, 0, 3,  0, 4, 2, 1,},
        {-5, -4, -5, 2, 0, 0,  0, 3, 2, 3,},
        {-6, -7, -4, 4, 4, 4,  0, 5, 2, 3,},
        {-5, -4, -5, 2, 0, 0,  0, 3, 2, 3,},
        {0,  0,  0,  3, 0, 3,  0, 4, 2, 1,},
        {4,  2,  1,  3, 0, 0,  0, 0, 1, 0,},
        {4,  2,  1,  0, 0, -1, 0, 1, 0, 0,}
};
EvalMatrix black_advisor_power_eval_matrix = {
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0,},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0,},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0,},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0,},
        {0, 0, 0, 0, 0, 0, 0, 0, 3, 0,},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0,},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0,},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0,},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0,}
};
EvalMatrix black_bishop_power_eval_matrix = {
        {0, 0, 0, 0, 0, 0, 0, -2, 0, 0,},
        {0, 0, 0, 0, 0, 0, 0, 0,  0, 0,},
        {0, 0, 0, 0, 0, 0, 0, 0,  0, 0,},
        {0, 0, 0, 0, 0, 0, 0, 0,  0, 0,},
        {0, 0, 0, 0, 0, 0, 0, 3,  0, 0,},
        {0, 0, 0, 0, 0, 0, 0, 0,  0, 0,},
        {0, 0, 0, 0, 0, 0, 0, 0,  0, 0,},
        {0, 0, 0, 0, 0, 0, 0, 0,  0, 0,},
        {0, 0, 0, 0, 0, 0, 0, -2, 0, 0,}
};
EvalMatrix black_pawn_power_eval_matrix = {
        {0, 20, 20, 20, 10, 3, -2, 0, 0, 0,},
        {0, 30, 30, 27, 18, 0, 0,  0, 0, 0,},
        {0, 50, 45, 30, 22, 4, -2, 0, 0, 0,},
        {2, 65, 55, 40, 35, 0, 0,  0, 0, 0,},
        {4, 70, 55, 42, 40, 7, 6,  0, 0, 0,},
        {2, 65, 55, 40, 35, 0, 0,  0, 0, 0,},
        {0, 50, 45, 30, 22, 4, -2, 0, 0, 0,},
        {0, 30, 30, 27, 18, 0, 0,  0, 0, 0,},
        {0, 20, 20, 20, 10, 3, -2, 0, 0, 0,}
};
EvalMatrix black_king_power_eval_matrix = {
        {0, 0, 0, 0, 0, 0, 0, 0,  0,  0,},
        {0, 0, 0, 0, 0, 0, 0, 0,  0,  0,},
        {0, 0, 0, 0, 0, 0, 0, 0,  0,  0,},
        {0, 0, 0, 0, 0, 0, 0, -9, -8, 1,},
        {0, 0, 0, 0, 0, 0, 0, -9, -8, 5,},
        {0, 0, 0, 0, 0, 0, 0, -9, -8, 1,},
        {0, 0, 0, 0, 0, 0, 0, 0,  0,  0,},
        {0, 0, 0, 0, 0, 0, 0, 0,  0,  0,},
        {0, 0, 0, 0, 0, 0, 0, 0,  0,  0,}
};

EvalMatrix empty_power_eval_matrix = {
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
};

std::map<ChessType, int> ChessValue = {
        {ChessType::RedRook,      500},
        {ChessType::RedKnight,    300},
        {ChessType::RedCannon,    300},
        {ChessType::RedAdvisor,   10},
        {ChessType::RedBishop,    30},
        {ChessType::RedPawn,      90},
        {ChessType::RedKing,      10000},
        {ChessType::BlackRook,    500},
        {ChessType::BlackKnight,  300},
        {ChessType::BlackCannon,  300},
        {ChessType::BlackAdvisor, 10},
        {ChessType::BlackBishop,  30},
        {ChessType::BlackPawn,    90},
        {ChessType::BlackKing,    10000},
        {ChessType::Empty,        0}
};

std::map<ChessType, int> MoveValue = {
        {ChessType::RedRook,      500},
        {ChessType::RedKnight,    100},
        {ChessType::RedCannon,    100},
        {ChessType::RedAdvisor,   0},
        {ChessType::RedBishop,    0},
        {ChessType::RedPawn,      -20},
        {ChessType::RedKing,      9999},
        {ChessType::BlackRook,    500},
        {ChessType::BlackKnight,  100},
        {ChessType::BlackCannon,  100},
        {ChessType::BlackAdvisor, 0},
        {ChessType::BlackBishop,  0},
        {ChessType::BlackPawn,    -20},
        {ChessType::BlackKing,    9999},
        {ChessType::Empty,        0}
};


/**
 * 棋子所在位置的评估（棋力评估）
 * @param type 棋子类型
 * @return 对应所有位置评估矩阵
 */
EvalMatrix getChessPowerEvalMatrix(ChessType type) {
    switch (type) {
        case ChessType::RedRook:
            return red_rook_power_eval_matrix;
        case ChessType::RedKnight:
            return red_knight_power_eval_matrix;
        case ChessType::RedCannon:
            return red_cannon_power_eval_matrix;
        case ChessType::RedAdvisor:
            return red_advisor_power_eval_matrix;
        case ChessType::RedBishop:
            return red_bishop_power_eval_matrix;
        case ChessType::RedPawn:
            return red_pawn_power_eval_matrix;
        case ChessType::RedKing:
            return red_king_power_eval_matrix;
        case ChessType::BlackRook:
            return black_rook_power_eval_matrix;
        case ChessType::BlackKnight:
            return black_knight_power_eval_matrix;
        case ChessType::BlackCannon:
            return black_cannon_power_eval_matrix;
        case ChessType::BlackAdvisor:
            return black_advisor_power_eval_matrix;
        case ChessType::BlackBishop:
            return black_bishop_power_eval_matrix;
        case ChessType::BlackPawn:
            return black_pawn_power_eval_matrix;
        case ChessType::BlackKing:
            return black_king_power_eval_matrix;
        case ChessType::Empty:
            std::cout << "Empty chess type has no evaluate" << std::endl;
            return empty_power_eval_matrix;
        default:
            std::cout << "Chess type not found" << std::endl;
            return empty_power_eval_matrix;
    }
}


/**
 * 获取棋子价值
 * @param type 棋子类型
 * @return 棋子价值
 */

int getChessValue(ChessType type) {
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
int getMoveValue(ChessType type) {
    if (MoveValue.find(type) == MoveValue.end()) {
        std::cout << "Chess type not found" << std::endl;
        return 0;
    }
    return MoveValue[type];
}

