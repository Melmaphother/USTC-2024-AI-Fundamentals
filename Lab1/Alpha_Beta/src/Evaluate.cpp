
#include "Evaluate.h"

Evaluate::Evaluate() {
    RedRookEvaluate = {
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
    RedKnightEvaluate = {
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
    RedCannonEvaluate = {
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
    RedAdvisorEvaluate = {
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
    RedBishopEvaluate = {
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
    RedPawnEvaluate = {
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
    RedKingEvaluate = {
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
    BlackRookEvaluate = getBlackEvaluateFromRed(RedRook);
    BlackKnightEvaluate = getBlackEvaluateFromRed(RedKnight);
    BlackCannonEvaluate = getBlackEvaluateFromRed(RedCannon);
    BlackAdvisorEvaluate = getBlackEvaluateFromRed(RedAdvisor);
    BlackBishopEvaluate = getBlackEvaluateFromRed(RedBishop);
    BlackPawnEvaluate = getBlackEvaluateFromRed(RedPawn);
    BlackKingEvaluate = getBlackEvaluateFromRed(RedKing);

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


std::vector<std::vector<int>> Evaluate::getBlackEvaluateFromRed(ChessType type) {
    std::vector<std::vector<int>> BlackEvaluate;
    auto RedEvaluate = getChessEvaluate(type);
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
std::vector<std::vector<int>> Evaluate::getChessEvaluate(ChessType type){
    switch (type) {
        case RedRook:
            return RedRookEvaluate;
        case RedKnight:
            return RedKnightEvaluate;
        case RedCannon:
            return RedCannonEvaluate;
        case RedAdvisor:
            return RedAdvisorEvaluate;
        case RedBishop:
            return RedBishopEvaluate;
        case RedPawn:
            return RedPawnEvaluate;
        case RedKing:
            return RedKingEvaluate;
        case BlackRook:
            return BlackRookEvaluate;
        case BlackKnight:
            return BlackKnightEvaluate;
        case BlackCannon:
            return BlackCannonEvaluate;
        case BlackAdvisor:
            return BlackAdvisorEvaluate;
        case BlackBishop:
            return BlackBishopEvaluate;
        case BlackPawn:
            return BlackPawnEvaluate;
        case BlackKing:
            return BlackKingEvaluate;
        case Empty:
            std::cout << "Empty chess type has no evaluate" << std::endl;
            return std::vector<std::vector<int>>(10, std::vector<int>(9, 0));
    }
}

/**
 * 获取棋盘上当前下棋方所有棋子的价值
 * @param chessboard 棋盘
 * @param color 当前下棋方
 * @return 棋盘上所有棋子的价值
 */

int Evaluate::getAllChessValueOfColor(std::vector<std::vector<ChessType>> &chessboard, ChessColor color) {
    int value = 0;
    for (int i = 0; i < chessboard.size(); i++) {
        for (int j = 0; j < chessboard[0].size(); j++) {
            ChessColor chess_color = isupper(chessboard[i][j]) ? Red : Black;
            if (chess_color == color) {
                value += ChessValue[chessboard[i][j]];
            }
        }
    }
    return value;
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
