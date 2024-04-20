#ifndef ALPHA_BETA_EVALUATE_H
#define ALPHA_BETA_EVALUATE_H

#include <iostream>
#include <vector>
#include <map>
#include "Chess.h"

extern std::vector<std::vector<int>> red_rook_power_eval_matrix;
extern std::vector<std::vector<int>> red_knight_power_eval_matrix;
extern std::vector<std::vector<int>> red_cannon_power_eval_matrix;
extern std::vector<std::vector<int>> red_advisor_power_eval_matrix;
extern std::vector<std::vector<int>> red_bishop_power_eval_matrix;
extern std::vector<std::vector<int>> red_pawn_power_eval_matrix;
extern std::vector<std::vector<int>> red_king_power_eval_matrix;

extern std::vector<std::vector<int>> black_rook_power_eval_matrix;
extern std::vector<std::vector<int>> black_knight_power_eval_matrix;
extern std::vector<std::vector<int>> black_cannon_power_eval_matrix;
extern std::vector<std::vector<int>> black_advisor_power_eval_matrix;
extern std::vector<std::vector<int>> black_bishop_power_eval_matrix;
extern std::vector<std::vector<int>> black_pawn_power_eval_matrix;
extern std::vector<std::vector<int>> black_king_power_eval_matrix;

extern std::vector<std::vector<int>> empty_power_eval_matrix;

// 棋子价值评估
extern std::map<ChessType, int> ChessValue;

// 行棋价值评估
extern std::map<ChessType, int> MoveValue;

// 行棋过程中
std::vector<std::vector<int>> getChessPowerEvalMatrix(ChessType type);

int getChessValue(ChessType type);

int getMoveValue(ChessType type);

// 初始化棋盘时的初始评估
int getAllChessPowerEval(std::vector<std::vector<ChessType>> &chessboard_matrix, ChessColor color);

int getAllChessValue(std::vector<std::vector<ChessType>> &chessboard_matrix, ChessColor color);


#endif //ALPHA_BETA_EVALUATE_H
