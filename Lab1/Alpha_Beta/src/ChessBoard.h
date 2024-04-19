
#ifndef ALPHA_BETA_CHESSBOARD_H
#define ALPHA_BETA_CHESSBOARD_H

#include <vector>
#include <string>
#include <map>
#include "Chess.h"
#include "Evaluate.h"

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
    std::vector<Chess> chesses; // 棋子
    std::vector<std::vector<ChessType>> chessboard; // 棋盘
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
        auto chessboard_from_file = getChessBoardFromFile(input_file);
        initChessBoard(chessboard_from_file, Red);
    }
    ChessBoard(std::vector<std::vector<ChessType>>& chessboard, ChessColor curr_color) {
        initChessBoard(chessboard, curr_color);
    }

private:
    static std::vector<std::vector<ChessType>> getChessBoardFromFile(const std::string &input_file);
    void initChessBoard(std::vector<std::vector<ChessType>>& chessboard, ChessColor curr_color);

public:
    std::vector<Chess> getChesses() const { return chesses; }
    std::vector<std::vector<ChessType>> getChessBoard() const { return chessboard; }
    std::vector<Move> getMoves() const { return moves; }
};

#endif //ALPHA_BETA_CHESSBOARD_H
