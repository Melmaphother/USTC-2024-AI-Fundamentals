#ifndef ALPHA_BETA_ALPHABETA_H
#define ALPHA_BETA_ALPHABETA_H

#include "ChessBoard.h"

int AlphaBetaSearch(ChessBoard &node, int depth, int alpha, int beta, bool isMaxNode);

Move getBestMoveFromChildren(ChessBoard &node, int depth, int root_score);

std::pair<int, Move> AlphaBetaMultiThreadSearch(ChessBoard &node, int depth, int alpha, int beta, bool isMaxNode);

void WriteMoveToFile(const std::string &output_file, const Move &move);


#endif //ALPHA_BETA_ALPHABETA_H
