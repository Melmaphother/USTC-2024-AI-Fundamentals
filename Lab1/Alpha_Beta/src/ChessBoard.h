
#ifndef ALPHA_BETA_CHESSBOARD_H
#define ALPHA_BETA_CHESSBOARD_H

#include <vector>
#include <string>
#include <map>
#include "Chess.h"
#include "Evaluate.h"

typedef std::vector<std::vector<ChessType>> ChessBoardMatrix;

// 动作结构体
struct Move {
    int init_x;
    int init_y;
    int next_x;
    int next_y;
    int score{0};
    bool is_eat{false}; // 是否吃子
    Move(int init_x, int init_y, int next_x, int next_y) : init_x(init_x), init_y(init_y), next_x(next_x), next_y(next_y) {}
};

// 棋盘类
class ChessBoard {
private:
    int width{9};
    int height{10};
    ChessBoardMatrix chessboard_matrix; // 棋盘
    ChessColor curr_color{Red}; // 当前下棋方

    std::vector<Move> moves; // 当前棋盘下可以走的动作

    Evaluate eval; // 评估

private:
    void getRookMoves(int x, int y);
    void getKnightMoves(int x, int y);
    void getCannonMoves(int x, int y);
    void getAdvisorMoves(int x, int y);
    void getBishopMoves(int x, int y);
    void getPawnMoves(int x, int y);
    void getKingMoves(int x, int y);

public:
    explicit ChessBoard(const std::string &input_file) {
        auto chessboard_matrix_from_file = getChessBoardMatrixFromFile(input_file);
        initChessBoard(chessboard_matrix_from_file, Red);
    }
    ChessBoard(ChessBoardMatrix& _chessboard_matrix, ChessColor _curr_color) {
        initChessBoard(_chessboard_matrix, _curr_color);
    }

private:
    static ChessBoardMatrix getChessBoardMatrixFromFile(const std::string &input_file);
    void initChessBoard(ChessBoardMatrix& _chessboard_matrix, ChessColor _curr_color);

public:
    ChessBoardMatrix getChessBoardMatrix() const { return chessboard_matrix; }
    std::vector<Move> getMoves() const { return moves; }
};

#endif //ALPHA_BETA_CHESSBOARD_H
