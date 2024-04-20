#ifndef ALPHA_BETA_EVALUATE_H
#define ALPHA_BETA_EVALUATE_H

#include <iostream>
#include <vector>
#include <map>
#include "Chess.h"

class Evaluate {
private:
    // 棋力评估矩阵
    std::vector<std::vector<int>> red_rook_power_eval_matrix;
    std::vector<std::vector<int>> red_knight_power_eval_matrix;
    std::vector<std::vector<int>> red_cannon_power_eval_matrix;
    std::vector<std::vector<int>> red_advisor_power_eval_matrix;
    std::vector<std::vector<int>> red_bishop_power_eval_matrix;
    std::vector<std::vector<int>> red_pawn_power_eval_matrix;
    std::vector<std::vector<int>> red_king_power_eval_matrix;

    std::vector<std::vector<int>> black_rook_power_eval_matrix;
    std::vector<std::vector<int>> black_knight_power_eval_matrix;
    std::vector<std::vector<int>> black_cannon_power_eval_matrix;
    std::vector<std::vector<int>> black_advisor_power_eval_matrix;
    std::vector<std::vector<int>> black_bishop_power_eval_matrix;
    std::vector<std::vector<int>> black_pawn_power_eval_matrix;
    std::vector<std::vector<int>> black_king_power_eval_matrix;

    // 棋子价值评估
    std::map<ChessType, int> ChessValue;

    // 行棋价值评估
    std::map<ChessType, int> MoveValue;

private:
    std::vector<std::vector<int>> getBlackEvalFromRed(ChessType type);

public:
    Evaluate();

    std::vector<std::vector<int>> getChessPowerEvalMatrix(ChessType type);
    int getChessValue(ChessType type);
    int getMoveValue(ChessType type);
};



#endif //ALPHA_BETA_EVALUATE_H
