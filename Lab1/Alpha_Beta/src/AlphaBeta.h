
#ifndef ALPHA_BETA_ALPHA_BETA_H
#define ALPHA_BETA_ALPHA_BETA_H

#include "GameTree.h"
#include <limits>

class AlphaBeta {
private:
    GameTreeNode root;
    std::vector<Move> best_moves;

public:
    AlphaBeta(ChessBoardMatrix &init_chessboard_matrix, int _max_depth) :
            root(Red, init_chessboard_matrix, _max_depth) {}

    int AlphaBetaSearch();
    std::vector<Move> getBestMoves();

private:
    int AlphaBetaSearch(GameTreeNode &node, int alpha, int beta);
    void getBestMoves(GameTreeNode &node);
};

#endif //ALPHA_BETA_ALPHA_BETA_H
