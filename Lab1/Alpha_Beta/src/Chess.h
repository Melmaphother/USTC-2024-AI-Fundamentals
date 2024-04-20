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


std::map<char, ChessType> getChessTypeFromChar = {
        {'R', RedRook},
        {'N', RedKnight},
        {'C', RedCannon},
        {'B', RedBishop},
        {'A', RedAdvisor},
        {'K', RedKing},
        {'P', RedPawn},
        {'r', BlackRook},
        {'n', BlackKnight},
        {'c', BlackCannon},
        {'b', BlackBishop},
        {'a', BlackAdvisor},
        {'k', BlackKing},
        {'p', BlackPawn},
        {'.', Empty}
};

inline ChessColor getChessColor(ChessType type) {
    if (type == Empty) {
        std::cerr << "Empty chess type has no color!" << std::endl;
    }
    return type >= 'A' && type <= 'Z' ? Red : Black;
}

#endif //ALPHA_BETA_CHESS_H
