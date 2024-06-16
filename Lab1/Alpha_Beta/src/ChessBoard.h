
#ifndef ALPHA_BETA_CHESSBOARD_H
#define ALPHA_BETA_CHESSBOARD_H

#include <vector>
#include <string>
#include <map>
#include "Chess.h"

using ChessBoardMatrix = std::vector<std::vector<ChessType>>;
#define END_GAME_SCORE (-1000000)


// 动作结构体
struct Move {
    ChessType chess_type;  // 走棋的棋子
    int init_x;
    int init_y;
    int next_x;
    int next_y;
    int score{0};
    bool is_eat{false}; // 是否吃子
    ChessType eat_chess_type{ChessType::Empty}; // 被吃的棋子
    Move(ChessType _chess_type, int _init_x, int _init_y, int _next_x, int _next_y) : chess_type(_chess_type), init_x(_init_x),
                                                                                 init_y(_init_y), next_x(_next_x),
                                                                                 next_y(_next_y) {}
};

// 棋盘类
class ChessBoard {
private:
    ChessBoardMatrix chessboard_matrix; // 棋盘
    ChessColor curr_color{ChessColor::Red}; // 当前下棋方
    bool is_stop_game{false}; // 是否结束游戏
    std::vector<Move> red_moves; // 红方所有可走的棋
    std::vector<Move> black_moves; // 黑方所有可走的棋

public:
    explicit ChessBoard(const std::string &input_file) ;

    int getCurrChessBoardScore();
    ChessBoard getChildChessBoardFromMove(const Move &move);

    ChessColor getCurrColor() const { return curr_color; }
    bool isStopGame() const { return is_stop_game; }
    std::vector<Move> getAllPossibleMoves() const { return curr_color == ChessColor::Red ? red_moves : black_moves; }

private:
    void updateMoves();

    void getRookMoves(int x, int y);
    void getKnightMoves(int x, int y);
    void getCannonMoves(int x, int y);
    void getAdvisorMoves(int x, int y);
    void getBishopMoves(int x, int y);
    void getPawnMoves(int x, int y);
    void getKingMoves(int x, int y);
};

#endif //ALPHA_BETA_CHESSBOARD_H
