#ifndef ALPHA_BETA_CHESS_H
#define ALPHA_BETA_CHESS_H

#include <map>
#include <iostream>

enum ChessColor {
    Red = true,
    Black = false
};

enum ChessType {
    RedRook = 'R',
    RedKnight = 'N',
    RedCannon = 'C',
    RedBishop = 'B',
    RedAdvisor = 'A',
    RedKing = 'K',
    RedPawn = 'P',
    BlackRook = 'r',
    BlackKnight = 'n',
    BlackCannon = 'c',
    BlackBishop = 'b',
    BlackAdvisor = 'a',
    BlackKing = 'k',
    BlackPawn = 'p',
    Empty = '.'
};

// c++17 之后这里可以加 inline 保证不会重复定义
extern std::map<char, ChessType> getChessTypeFromChar;

extern std::map<ChessType, std::string> getChessStringFromType;

inline ChessColor getChessColor(ChessType type) {
    if (type == Empty) {
        std::cerr << "Empty chess type has no color!" << std::endl;
    }
    return type >= 'A' && type <= 'Z' ? Red : Black;
}

#endif //ALPHA_BETA_CHESS_H
