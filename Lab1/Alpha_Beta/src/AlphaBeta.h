#ifndef ALPHA_BETA_ALPHABETA_H
#define ALPHA_BETA_ALPHABETA_H

#include "ChessBoard.h"

int alphaBetaSearch(ChessBoard &node, int depth, int alpha, int beta, bool isMaxNode);

Move getBestMoveFromChildren(ChessBoard &node, int depth, int root_score);

std::pair<int, Move> alphaBetaSearchMultiThreads(ChessBoard &node, int depth, int alpha, int beta, bool isMaxNode);

void writeMoveToFile(const std::string &output_file, const Move &move);


#endif //ALPHA_BETA_ALPHABETA_H
