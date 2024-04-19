//
// Created by 15597 on 24-4-19.
//

#ifndef ALPHA_BETA_EVALUATE_H
#define ALPHA_BETA_EVALUATE_H

#include <iostream>
#include <vector>
#include <map>
#include "Chess.h"

class Evaluate {
private:
    // 棋力评估
    std::vector<std::vector<int>> RedRookEvaluate;
    std::vector<std::vector<int>> RedKnightEvaluate;
    std::vector<std::vector<int>> RedCannonEvaluate;
    std::vector<std::vector<int>> RedAdvisorEvaluate;
    std::vector<std::vector<int>> RedBishopEvaluate;
    std::vector<std::vector<int>> RedPawnEvaluate;
    std::vector<std::vector<int>> RedKingEvaluate;

    std::vector<std::vector<int>> BlackRookEvaluate;
    std::vector<std::vector<int>> BlackKnightEvaluate;
    std::vector<std::vector<int>> BlackCannonEvaluate;
    std::vector<std::vector<int>> BlackAdvisorEvaluate;
    std::vector<std::vector<int>> BlackBishopEvaluate;
    std::vector<std::vector<int>> BlackPawnEvaluate;
    std::vector<std::vector<int>> BlackKingEvaluate;

    // 棋子价值评估
    static std::map<ChessType, int> ChessValue;

    // 行棋价值评估
    std::map<ChessType, int> MoveValue;

private:
    std::vector<std::vector<int>> getBlackEvaluateFromRed(ChessType type);

public:
    Evaluate();

    std::vector<std::vector<int>> getChessEvaluate(ChessType type);
    static int getAllChessValueOfColor(std::vector<std::vector<ChessType>> &chessboard, ChessColor color);
    int getMoveValue(ChessType type);
};



#endif //ALPHA_BETA_EVALUATE_H
