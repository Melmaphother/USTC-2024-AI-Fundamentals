
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
    ChessType chess_type;  // 走棋的棋子
    int init_x;
    int init_y;
    int next_x;
    int next_y;
    int score{0};
    bool is_eat{false}; // 是否吃子
    ChessType eat_chess_type{Empty}; // 被吃的棋子
    Move(ChessType _chess_type, int _init_x, int _init_y, int _next_x, int _next_y) : chess_type(_chess_type), init_x(_init_x),
                                                                                 init_y(_init_y), next_x(_next_x),
                                                                                 next_y(_next_y) {}
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

    ChessBoard(ChessBoardMatrix &_chessboard_matrix, ChessColor _curr_color) {
        initChessBoard(_chessboard_matrix, _curr_color);
    }

private:
    static ChessBoardMatrix getChessBoardMatrixFromFile(const std::string &input_file);

    void initChessBoard(ChessBoardMatrix &_chessboard_matrix, ChessColor _curr_color);

public:
    ChessBoardMatrix getChessBoardMatrix() const { return chessboard_matrix; }

    std::vector<Move> getMoves() const { return moves; }
};

#endif //ALPHA_BETA_CHESSBOARD_H
