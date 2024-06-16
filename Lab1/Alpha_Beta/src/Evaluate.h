#ifndef ALPHA_BETA_EVALUATE_H
#define ALPHA_BETA_EVALUATE_H

#include <iostream>
#include <vector>
#include <map>
#include "Chess.h"

using EvalMatrix = std::vector<std::vector<int>>;

extern EvalMatrix red_rook_power_eval_matrix;
extern EvalMatrix red_knight_power_eval_matrix;
extern EvalMatrix red_cannon_power_eval_matrix;
extern EvalMatrix red_advisor_power_eval_matrix;
extern EvalMatrix red_bishop_power_eval_matrix;
extern EvalMatrix red_pawn_power_eval_matrix;
extern EvalMatrix red_king_power_eval_matrix;

extern EvalMatrix black_rook_power_eval_matrix;
extern EvalMatrix black_knight_power_eval_matrix;
extern EvalMatrix black_cannon_power_eval_matrix;
extern EvalMatrix black_advisor_power_eval_matrix;
extern EvalMatrix black_bishop_power_eval_matrix;
extern EvalMatrix black_pawn_power_eval_matrix;
extern EvalMatrix black_king_power_eval_matrix;

extern EvalMatrix empty_power_eval_matrix;

// 棋子价值评估
extern std::map<ChessType, int> ChessValue;

// 行棋价值评估
extern std::map<ChessType, int> MoveValue;

// 行棋过程中
EvalMatrix getChessPowerEvalMatrix(ChessType type);

int getChessValue(ChessType type);

int getMoveValue(ChessType type);


#endif //ALPHA_BETA_EVALUATE_H
