#include "ChessBoard.h"
#include <fstream>

void ChessBoard::initChessBoard(std::vector<std::vector<ChessType>> &_chessboard, ChessColor _curr_color) {
    this->chessboard = _chessboard;
    this->curr_color = _curr_color;
    this->width = static_cast<int>(chessboard.size());
    this->height = static_cast<int>(chessboard[0].size());

    // 生成所有合法的走法
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            ChessType single_chess_type = this->chessboard[i][j];
            if (single_chess_type == Empty) continue; // 空位置不需要考虑

            ChessColor single_chess_color = getChessColor(single_chess_type);

            if (single_chess_color != curr_color) continue;  // 不是当前下棋方的棋子，不需要考虑

            switch (std::tolower(single_chess_type)) {
                case 'r':
                    getRookMoves(i, j);
                    break;
                case 'n':
                    getKnightMoves(i, j);
                    break;
                case 'c':
                    getCannonMoves(i, j);
                    break;
                case 'a':
                    getAdvisorMoves(i, j);
                    break;
                case 'b':
                    getBishopMoves(i, j);
                    break;
                case 'p':
                    getPawnMoves(i, j);
                    break;
                case 'k':
                    getKingMoves(i, j);
                    break;
                default:
                    std::cout << "Invalid chess type" << std::endl;
            }
        }
    }

}

std::vector<std::vector<ChessType>> ChessBoard::getChessBoardFromFile(const std::string &input_file) {
    std::vector<std::vector<ChessType>> chessboard_from_file;
    // chessboard的大小应当为 9 * 10
    chessboard_from_file.resize(9, std::vector<ChessType>(10));
    std::ifstream file(input_file);
    std::string line;
    int curr_height = 9;
    while (std::getline(file, line)) {
        for (int i = 0; i < line.size(); i++) {
            chessboard_from_file[i][curr_height] = getChessTypeFromChar[line[i]];
        }
        curr_height--;
        if (curr_height < 0) break;
    }
    file.close();
    return chessboard_from_file;
}


void ChessBoard::getRookMoves(int x, int y) {
    //前后左右分别进行搜索，遇到棋子停止，不同阵营可以吃掉
    std::vector<Move> RookMoves;
    for (int i = x + 1; i < width; i++){
        Move move(x, y, i, y);
        if (chessboard[i][y] != Empty) {
            ChessColor obstacle_color = getChessColor(chessboard[i][y]);
            if (obstacle_color != curr_color) {
                // 车可以吃掉对方的棋子
                move.score = eval.getMoveValue(chessboard[i][y]);
                move.is_eat = true;
                RookMoves.push_back(move);
            }
            break;
        }
        RookMoves.push_back(move);
    }

    for (int i = x - 1; i >= 0; i--){
        Move move(x, y, i, y);
        if (chessboard[i][y] != Empty) {
            ChessColor obstacle_color = getChessColor(chessboard[i][y]);
            if (obstacle_color != curr_color) {
                move.score = eval.getMoveValue(chessboard[i][y]);
                move.is_eat = true;
                RookMoves.push_back(move);
            }
            break;
        }
        RookMoves.push_back(move);
    }

    for (int j = y + 1; j < height; j++){
        Move move(x, y, x, j);
        if (chessboard[x][j] != Empty) {
            ChessColor obstacle_color = getChessColor(chessboard[x][j]);
            if (obstacle_color != curr_color) {
                move.score = eval.getMoveValue(chessboard[x][j]);
                move.is_eat = true;
                RookMoves.push_back(move);
            }
            break;
        }
        RookMoves.push_back(move);
    }

    for (int j = y - 1; j >= 0; j--){
        Move move(x, y, x, j);
        if (chessboard[x][j] != Empty) {
            ChessColor obstacle_color = getChessColor(chessboard[x][j]);
            if (obstacle_color != curr_color) {
                move.score = eval.getMoveValue(chessboard[x][j]);
                move.is_eat = true;
                RookMoves.push_back(move);
            }
            break;
        }
        RookMoves.push_back(move);
    }

    for (auto &move : RookMoves) {
        // 加入棋力变化，注意这里需要使用init位置的Chess评估矩阵，因为目前还没有移动，next的位置还是对方的棋子
        auto chess_eval_matrix = eval.getChessPowerEvalMatrix(chessboard[move.init_x][move.init_y]);
        move.score += chess_eval_matrix[move.next_x][move.next_y] - chess_eval_matrix[move.init_x][move.init_y];
        moves.push_back(move);
    }
}

void ChessBoard::getKnightMoves(int x, int y) {
    std::vector<Move> KnightMoves;
    int dx[] = {-1, 1, 2, 2, 1, -1, -2, -2};
    int dy[] = {2, 2, 1, -1, -2, -2, -1, 1};
    int block_x[] = {0, 0, 1, 1, 0, 0, -1, -1};
    int block_y[] = {1, 1, 0, 0, -1, -1, 0, 0}; // 蹩马腿
    for(int i = 0; i < 8; i++){
        int next_x = x + dx[i];
        int next_y = y + dy[i];
        // 跳出棋盘，丢弃
        if (next_x < 0 || next_x >8 || next_y < 0 || next_y > 9) continue;
        // 蹩马腿，丢弃
        if (chessboard[x + block_x[i]][y + block_y[i]] != Empty) continue;

        Move move(x, y, next_x, next_y);
        if (chessboard[next_x][next_y] != Empty) {
            ChessColor obstacle_color = getChessColor(chessboard[next_x][next_y]);
            if (obstacle_color != curr_color) {
                // 马可以吃掉对方的棋子
                move.score = eval.getMoveValue(chessboard[next_x][next_y]);
                move.is_eat = true;
                KnightMoves.push_back(move);
            }
            // 遇到棋子，如果是己方的棋子，不能走
            continue;
        }
        KnightMoves.push_back(move);
    }
    for (auto &move : KnightMoves) {
        // 加入棋力变化
        auto chess_eval_matrix = eval.getChessPowerEvalMatrix(chessboard[move.init_x][move.init_y]);
        move.score += chess_eval_matrix[move.next_x][move.next_y] - chess_eval_matrix[move.init_x][move.init_y];
        moves.push_back(move);
    }
}

void ChessBoard::getCannonMoves(int x, int y) {
    // 炮与车的走法类似，但是吃子时需要隔一个子
    // 炮吃子的条件：从炮当前位置的下一个位置开始，搜索到第一个棋子，无论是己方还是对方的棋子，继续搜索，如果遇到第二个棋子，且是对方的棋子，则可以吃掉
    // 炮能走的位置：类似车的位置 + 吃子的位置
    std::vector<Move> CannonMoves;
    for (int i = x + 1; i < width; i++){
        Move move(x, y, i, y);
        if (chessboard[i][y] != Empty) {
            // 碰到第一个棋子，无论颜色，继续搜索
            for (int k = i + 1; k < width; k++) {
                if (chessboard[k][y] != Empty) {
                    // 碰到第二个棋子，若是对方的棋子，则可以吃掉
                    ChessColor obstacle_color = getChessColor(chessboard[k][y]);
                    if (obstacle_color != curr_color) {
                        // 炮可以吃掉对方的棋子
                        move.score = eval.getMoveValue(chessboard[k][y]);
                        move.is_eat = true;
                        CannonMoves.push_back(move);
                    }
                    break;
                }
            }
            // 无论是否打到对方的棋子，都要跳出外层循环
            break;
        }
        CannonMoves.push_back(move);
    }

    for (int i = x - 1; i >= 0; i--){
        Move move(x, y, i, y);
        if (chessboard[i][y] != Empty) {
            for (int k = i - 1; k >= 0; k--) {
                if (chessboard[k][y] != Empty) {
                    ChessColor obstacle_color = getChessColor(chessboard[k][y]);
                    if (obstacle_color != curr_color) {
                        move.score = eval.getMoveValue(chessboard[k][y]);
                        move.is_eat = true;
                        CannonMoves.push_back(move);
                    }
                    break;
                }
            }
            break;
        }
        CannonMoves.push_back(move);
    }

    for (int j = y + 1; j < height; j++){
        Move move(x, y, x, j);
        if (chessboard[x][j] != Empty) {
            for (int k = j + 1; k < height; k++) {
                if (chessboard[x][k] != Empty) {
                    ChessColor obstacle_color = getChessColor(chessboard[x][k]);
                    if (obstacle_color != curr_color) {
                        move.score = eval.getMoveValue(chessboard[x][k]);
                        move.is_eat = true;
                        CannonMoves.push_back(move);
                    }
                    break;
                }
            }
            break;
        }
        CannonMoves.push_back(move);
    }

    for (int j = y - 1; j >= 0; j--){
        Move move(x, y, x, j);
        if (chessboard[x][j] != Empty) {
            for (int k = j - 1; k >= 0; k--) {
                if (chessboard[x][k] != Empty) {
                    ChessColor obstacle_color = getChessColor(chessboard[x][k]);
                    if (obstacle_color != curr_color) {
                        move.score = eval.getMoveValue(chessboard[x][k]);
                        move.is_eat = true;
                        CannonMoves.push_back(move);
                    }
                    break;
                }
            }
            break;
        }
        CannonMoves.push_back(move);
    }

    for (auto &move : CannonMoves) {
        auto chess_eval_matrix = eval.getChessPowerEvalMatrix(chessboard[move.init_x][move.init_y]);
        move.score += chess_eval_matrix[move.next_x][move.next_y] - chess_eval_matrix[move.init_x][move.init_y];
        moves.push_back(move);
    }
}

void ChessBoard::getAdvisorMoves(int x, int y) {
    // 士的走法：走斜线，每次走一步，不能出九宫格
    // 红士的九宫格：3 <= x <= 5, 0 <= y <= 2
    // 黑士的九宫格：3 <= x <= 5, 7 <= y <= 9
    std::vector<Move> AdvisorMoves;
    int dx[] = {1, 1, -1, -1};
    int dy[] = {1, -1, 1, -1};
    for (int i = 0; i < 4; i++) {
        int next_x = x + dx[i];
        int next_y = y + dy[i];
        if (curr_color == Red) {
            if (next_x < 3 || next_x > 5 || next_y < 0 || next_y > 2) continue;
        } else {
            if (next_x < 3 || next_x > 5 || next_y < 7 || next_y > 9) continue;
        }

        Move move(x, y, next_x, next_y);
        if (chessboard[next_x][next_y] != Empty) {
            ChessColor obstacle_color = getChessColor(chessboard[next_x][next_y]);
            if (obstacle_color != curr_color) {
                move.score = eval.getMoveValue(chessboard[next_x][next_y]);
                move.is_eat = true;
                AdvisorMoves.push_back(move);
            }
            // 遇到棋子，如果是己方的棋子，不能走
            continue;
        }
        AdvisorMoves.push_back(move);
    }
    
    for (auto &move : AdvisorMoves) {
        auto chess_eval_matrix = eval.getChessPowerEvalMatrix(chessboard[move.init_x][move.init_y]);
        move.score += chess_eval_matrix[move.next_x][move.next_y] - chess_eval_matrix[move.init_x][move.init_y];
        moves.push_back(move);
    }
}

void ChessBoard::getBishopMoves(int x, int y) {
    // 象的走法：走田字，每次走两步，不能过河，并且不能被塞象眼
    // 红象的范围：0 <= x <= 8, 0 <= y <= 4
    // 黑象的范围：0 <= x <= 8, 5 <= y <= 9
    int dx[] = {2, 2, -2, -2};
    int dy[] = {2, -2, 2, -2};
    int block_x[] = {1, 1, -1, -1};
    int block_y[] = {1, -1, 1, -1};
    std::vector<Move> BishopMoves;
    for (int i = 0; i < 4; i++) {
        int next_x = x + dx[i];
        int next_y = y + dy[i];
        if (curr_color == Red) {
            if (next_x < 0 || next_x > 8 || next_y < 0 || next_y > 4) continue;
        } else {
            if (next_x < 0 || next_x > 8 || next_y < 5 || next_y > 9) continue;
        }

        Move move(x, y, next_x, next_y);
        if (chessboard[next_x][next_y] != Empty) continue;
        if (chessboard[x + block_x[i]][y + block_y[i]] != Empty) continue;
        if (chessboard[next_x][next_y] != Empty) {
            ChessColor obstacle_color = getChessColor(chessboard[next_x][next_y]);
            if (obstacle_color != curr_color) {
                move.score = eval.getMoveValue(chessboard[next_x][next_y]);
                move.is_eat = true;
                BishopMoves.push_back(move);
            }
            // 遇到棋子，如果是己方的棋子，不能走
            continue;
        }
        BishopMoves.push_back(move);
    }
    
    for (auto &move : BishopMoves) {
        auto chess_eval_matrix = eval.getChessPowerEvalMatrix(chessboard[move.init_x][move.init_y]);
        move.score += chess_eval_matrix[move.next_x][move.next_y] - chess_eval_matrix[move.init_x][move.init_y];
        moves.push_back(move);
    }
}

void ChessBoard::getPawnMoves(int x, int y) {
    // 卒的走法：红卒向上走，黑卒向下走，每次走一步，过河后可以左右走
    // 红卒的未过河位置：3 <= y <= 4，只能向 y 增大的方向走
    // 红卒的过河位置：5 <= y <= 9
    // 黑卒的未过河位置：5 <= y <= 6，只能向 y 减小的方向走
    // 黑卒的过河位置：0 <= y <= 4
    std::vector<Move> PawnMoves;
    if (curr_color == Red) {
        // 红卒未过河
        if (y >= 3 && y <= 4) {
            int next_y = y + 1;
            Move move(x, y, x, next_y);
            if (chessboard[x][next_y] != Empty) {
                ChessColor obstacle_color = getChessColor(chessboard[x][next_y]);
                if (obstacle_color != curr_color) {
                    move.score = eval.getMoveValue(chessboard[x][next_y]);
                    move.is_eat = true;
                    PawnMoves.push_back(move);
                }
            } else {
                PawnMoves.push_back(move);
            }
        } else if (y >= 5 && y <= 9) { // 红卒过河
            int dx[] = {-1, 1, 0};
            int dy[] = {0, 0, 1};
            for (int i = 0; i < 3; i++) {
                int next_x = x + dx[i];
                int next_y = y + dy[i];
                if (next_x < 0 || next_x > 8 || next_y < 5 || next_y > 9) continue;
                Move move(x, y, next_x, next_y);
                if (chessboard[next_x][next_y] != Empty) {
                    ChessColor obstacle_color = getChessColor(chessboard[next_x][next_y]);
                    if (obstacle_color != curr_color) {
                        move.score = eval.getMoveValue(chessboard[next_x][next_y]);
                        move.is_eat = true;
                        PawnMoves.push_back(move);
                    }
                    continue;
                }
                PawnMoves.push_back(move);
            }
        } else {
            std::cout << "Invalid red pawn position" << std::endl;
        }
    } else {
        // 黑卒未过河
        if (y >= 5 && y <= 6) {
            int next_y = y - 1;
            Move move(x, y, x, next_y);
            if (chessboard[x][next_y] != Empty) {
                ChessColor obstacle_color = getChessColor(chessboard[x][next_y]);
                if (obstacle_color != curr_color) {
                    move.score = eval.getMoveValue(chessboard[x][next_y]);
                    move.is_eat = true;
                    PawnMoves.push_back(move);
                }
            } else {
                PawnMoves.push_back(move);
            }
        } else if (y >= 0 && y <= 4) { // 黑卒过河
            int dx[] = {-1, 1, 0};
            int dy[] = {0, 0, -1};
            for (int i = 0; i < 3; i++) {
                int next_x = x + dx[i];
                int next_y = y + dy[i];
                if (next_x < 0 || next_x > 8 || next_y < 0 || next_y > 4) continue;

                Move move(x, y, next_x, next_y);
                if (chessboard[next_x][next_y] != Empty) {
                    ChessColor obstacle_color = getChessColor(chessboard[next_x][next_y]);
                    if (obstacle_color != curr_color) {
                        move.score = eval.getMoveValue(chessboard[next_x][next_y]);
                        move.is_eat = true;
                        PawnMoves.push_back(move);
                    }
                    continue;
                }
                PawnMoves.push_back(move);
            }
        } else {
            std::cout << "Invalid black pawn position" << std::endl;
        }
    }

    for (auto &move : PawnMoves) {
        auto chess_eval_matrix = eval.getChessPowerEvalMatrix(chessboard[move.init_x][move.init_y]);
        move.score += chess_eval_matrix[move.next_x][move.next_y] - chess_eval_matrix[move.init_x][move.init_y];
        moves.push_back(move);
    }
}

void ChessBoard::getKingMoves(int x, int y) {
    // 将的走法：走直线，每次走一步，不能出九宫格
    // 红帅的九宫格：3 <= x <= 5, 0 <= y <= 2
    // 黑将的九宫格：3 <= x <= 5, 7 <= y <= 9
    // 将帅不能直接对面，即不能在同一列上，为了表示棋局的结束，认为当将帅对面时，本方可以将对方将/帅吃掉
    std::vector<Move> KingMoves;
    // 首先考虑将帅直接对面的情况
    if (curr_color == Red) {
        for (int j = y + 1; j < height; j++) {
            if (chessboard[x][j] != Empty) {
                if (chessboard[x][j] == BlackKing) {
                    Move move(x, y, x, j);
                    move.score = eval.getMoveValue(chessboard[x][j]);
                    move.is_eat = true;
                    KingMoves.push_back(move);
                }
                break;
            }
        }
    } else {
        for (int j = y - 1; j >= 0; j--) {
            if (chessboard[x][j] != Empty) {
                if (chessboard[x][j] == RedKing) {
                    Move move(x, y, x, j);
                    move.score = eval.getMoveValue(chessboard[x][j]);
                    move.is_eat = true;
                    KingMoves.push_back(move);
                }
                break;
            }
        }
    }

    int dx[] = {1, 1, -1, -1};
    int dy[] = {1, -1, 1, -1};
    for (int i = 0; i < 4; i++) {
        int next_x = x + dx[i];
        int next_y = y + dy[i];
        if (curr_color == Red) {
            if (next_x < 3 || next_x > 5 || next_y < 0 || next_y > 2) continue;
        } else {
            if (next_x < 3 || next_x > 5 || next_y < 7 || next_y > 9) continue;
        }

        Move move(x, y, next_x, next_y);
        if (chessboard[next_x][next_y] != Empty) {
            ChessColor obstacle_color = getChessColor(chessboard[next_x][next_y]);
            if (obstacle_color != curr_color) {
                move.score = eval.getMoveValue(chessboard[next_x][next_y]);
                move.is_eat = true;
                KingMoves.push_back(move);
            }
            // 遇到棋子，如果是己方的棋子，不能走
            continue;
        }
        KingMoves.push_back(move);
    }

    for (auto &move : KingMoves) {
        auto chess_eval_matrix = eval.getChessPowerEvalMatrix(chessboard[move.init_x][move.init_y]);
        move.score += chess_eval_matrix[move.next_x][move.next_y] - chess_eval_matrix[move.init_x][move.init_y];
        moves.push_back(move);
    }
}




