#ifndef ALPHA_BETA_CHESS_H
#define ALPHA_BETA_CHESS_H

#include <map>
#include <iostream>

enum class ChessColor {
    Red = true,
    Black = false
};

enum class ChessType {
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

inline ChessColor getChessColor(ChessType type) {
    if (type == ChessType::Empty) {
        std::cerr << "Empty chess type has no color!" << std::endl;
    }
    auto chess_char = static_cast<char>(type);
    return chess_char >= 'A' && chess_char <= 'Z' ? ChessColor::Red : ChessColor::Black;
}

#endif //ALPHA_BETA_CHESS_H
