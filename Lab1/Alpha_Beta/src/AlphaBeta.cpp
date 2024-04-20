#include "AlphaBeta.h"


int AlphaBeta::AlphaBetaSearch() {
    return AlphaBetaSearch(root, std::numeric_limits<int>::min(), std::numeric_limits<int>::max());
}

int AlphaBeta::AlphaBetaSearch(GameTreeNode &node, int alpha, int beta) {
    if (node.isLeafNode()) {
        int node_score = node.getNodeScore();
        node.setNodeSearchScore(node_score);
        return node_score;
    }

    if (node.isMaxNode()) {
        int max_score = std::numeric_limits<int>::min();
        for (auto &child: node.getChildren()) {
            int score = AlphaBetaSearch(child, alpha, beta);
            max_score = std::max(max_score, score);
            alpha = std::max(alpha, max_score);
            if (beta <= alpha) {
                break;
            }
        }
        node.setNodeSearchScore(max_score);
        return max_score;
    } else {
        int min_score = std::numeric_limits<int>::max();
        for (auto &child: node.getChildren()) {
            int score = AlphaBetaSearch(child, alpha, beta);
            min_score = std::min(min_score, score);
            beta = std::min(beta, min_score);
            if (beta <= alpha) {
                break;
            }
        }
        node.setNodeSearchScore(min_score);
        return min_score;
    }
}

std::vector<Move> AlphaBeta::getBestMoves() {
    getBestMoves(root);
    return best_moves;
}

void AlphaBeta::getBestMoves(GameTreeNode &node) {
    if (node.isLeafNode()) {
        // 最佳路径搜索完毕
        best_moves.push_back(node.getParentToChildMove());
        return;
    }
    // 寻找 node 子结点中搜索分数与 node 的搜索分数相等的结点，递归直到找到叶子结点
    for (auto &child: node.getChildren()) {
        if (child.getNodeSearchScore() == node.getNodeSearchScore()) {
            best_moves.push_back(child.getParentToChildMove());
            getBestMoves(child);
            break;
        }
    }
}