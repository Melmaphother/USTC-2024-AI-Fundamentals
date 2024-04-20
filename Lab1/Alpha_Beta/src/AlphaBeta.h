
#ifndef ALPHA_BETA_ALPHA_BETA_H
#define ALPHA_BETA_ALPHA_BETA_H

#include "GameTree.h"

class AlphaBeta {
private:
    GameTreeNode root;
    int max_depth{10};
    
public:
    AlphaBeta(ChessBoardMatrix _init_chessboard_matrix, int _max_depth) : 
        root(GameTreeNode(Red, _init_chessboard_matrix, _max_depth)),
        max_depth(_max_depth) {}
};

#endif //ALPHA_BETA_ALPHA_BETA_H
