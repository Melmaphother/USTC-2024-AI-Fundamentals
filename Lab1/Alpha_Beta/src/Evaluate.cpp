#include "Evaluate.h"

std::vector<std::vector<int>> red_rook_power_eval_matrix = {
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
std::vector<std::vector<int>> red_knight_power_eval_matrix = {
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
std::vector<std::vector<int>> red_cannon_power_eval_matrix = {
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
std::vector<std::vector<int>> red_advisor_power_eval_matrix = {
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
std::vector<std::vector<int>> red_bishop_power_eval_matrix = {
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
std::vector<std::vector<int>> red_pawn_power_eval_matrix = {
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
std::vector<std::vector<int>> red_king_power_eval_matrix = {
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
std::vector<std::vector<int>> black_rook_power_eval_matrix = {
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
std::vector<std::vector<int>> black_knight_power_eval_matrix = {
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
std::vector<std::vector<int>> black_cannon_power_eval_matrix = {
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
std::vector<std::vector<int>> black_advisor_power_eval_matrix = {
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
std::vector<std::vector<int>> black_bishop_power_eval_matrix = {
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
std::vector<std::vector<int>> black_pawn_power_eval_matrix = {
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
std::vector<std::vector<int>> black_king_power_eval_matrix = {
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

std::vector<std::vector<int>> empty_power_eval_matrix = {
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
        {RedRook,      500},
        {RedKnight,    300},
        {RedCannon,    300},
        {RedAdvisor,   10},
        {RedBishop,    30},
        {RedPawn,      90},
        {RedKing,      10000},
        {BlackRook,    500},
        {BlackKnight,  300},
        {BlackCannon,  300},
        {BlackAdvisor, 10},
        {BlackBishop,  30},
        {BlackPawn,    90},
        {BlackKing,    10000},
        {Empty,        0}
};

std::map<ChessType, int> MoveValue = {
        {RedRook,      500},
        {RedKnight,    100},
        {RedCannon,    100},
        {RedAdvisor,   0},
        {RedBishop,    0},
        {RedPawn,      -20},
        {RedKing,      9999},
        {BlackRook,    500},
        {BlackKnight,  100},
        {BlackCannon,  100},
        {BlackAdvisor, 0},
        {BlackBishop,  0},
        {BlackPawn,    -20},
        {BlackKing,    9999},
        {Empty,        0}
};


/**
 * 棋子所在位置的评估（棋力评估）
 * @param type 棋子类型
 * @return 对应所有位置评估矩阵
 */
std::vector<std::vector<int>> getChessPowerEvalMatrix(ChessType type) {
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

