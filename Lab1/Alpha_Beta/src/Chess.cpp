#include "Chess.h"

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

std::map<ChessType, char> getCharFromChessType = {
        {RedRook,      'R'},
        {RedKnight,    'N'},
        {RedCannon,    'C'},
        {RedBishop,    'B'},
        {RedAdvisor,   'A'},
        {RedKing,      'K'},
        {RedPawn,      'P'},
        {BlackRook,    'r'},
        {BlackKnight,  'n'},
        {BlackCannon,  'c'},
        {BlackBishop,  'b'},
        {BlackAdvisor, 'a'},
        {BlackKing,    'k'},
        {BlackPawn,    'p'},
        {Empty,        '.'}

};

std::map<ChessType, std::string> getChessNameFromType = {
        {RedRook,      "Red Rook"},
        {RedKnight,    "Red Knight"},
        {RedCannon,    "Red Cannon"},
        {RedBishop,    "Red Bishop"},
        {RedAdvisor,   "Red Advisor"},
        {RedKing,      "Red King"},
        {RedPawn,      "Red Pawn"},
        {BlackRook,    "Black Rook"},
        {BlackKnight,  "Black Knight"},
        {BlackCannon,  "Black Cannon"},
        {BlackBishop,  "Black Bishop"},
        {BlackAdvisor, "Black Advisor"},
        {BlackKing,    "Black King"},
        {BlackPawn,    "Black Pawn"},
        {Empty,        "Empty"}
};