#include "ChessBoard.h"
#include "Evaluate.h"
#include <fstream>


ChessBoard::ChessBoard(const std::string &input_file) {
    // chessboard的大小应当为 9 * 10
    chessboard_matrix.resize(9, std::vector<ChessType>(10));
    std::ifstream file(input_file);
    if (!file.is_open()) {
        std::cout << "Open file failed" << std::endl;
        return;
    }
    std::string line;
    int curr_height = 9;
    while (std::getline(file, line)) {
        for (int i = 0; i < line.size(); i++) {
            chessboard_matrix[i][curr_height] = getChessTypeFromChar[line[i]];
        }
        curr_height--;
        if (curr_height < 0) break;
    }
    file.close();
    // 更新所有合法的走法
    updateMoves();
}

void ChessBoard::updateMoves() {
    // 首先清空之前的走法
    moves.clear();
    // 生成所有合法的走法
    for (int i = 0; i < 9; i++) {
        for (int j = 0; j < 10; j++) {
            ChessType single_chess_type = chessboard_matrix[i][j];
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

    // 特殊的困毙情况
    if (moves.empty()) {
        is_stop_game = true;
    }
}

int ChessBoard::getCurrChessBoardScore() {
    // 获取当前棋盘的分数，为程序方的分数减去对方的分数
    // 当前方为红方则程序方为当前方，当前方为黑方则程序方为对方
    // 当前方的分数由三部分组成
    // 棋子位置代表的棋力 + 棋子自身的价值 + 所有 move 的分数之和
    // 对手方的分数由两部分组成
    // 棋子位置代表的棋力 + 棋子自身的价值
    int all_move_score = 0;
    for (const auto &move: moves) {
        all_move_score += move.score;
    }

    int curr_chess_power = 0;
    int opponent_chess_power = 0;
    int curr_value = 0;
    int opponent_value = 0;
    for (int i = 0; i < 9; i++) {
        for (int j = 0; j < 10; j++) {
            auto chess_type = chessboard_matrix[i][j];
            if (chess_type == Empty) continue;
            auto eval_matrix = getChessPowerEvalMatrix(chess_type);
            auto chess_color = getChessColor(chess_type);
            if (chess_color == curr_color) {
                curr_chess_power += eval_matrix[i][j];
                curr_value += getChessValue(chess_type);
            } else {
                opponent_chess_power += eval_matrix[i][j];
                opponent_value += getChessValue(chess_type);
            }
        }
    }
    int curr_score = curr_chess_power + curr_value + all_move_score;
    int opponent_score = opponent_chess_power + opponent_value;

    return curr_color == Red ? curr_score - opponent_score : opponent_score - curr_score;
}

ChessBoard ChessBoard::getChildChessBoardFromMove(const Move &move) {
    ChessBoard child_chessboard = *this;
    // 子结点行棋方交换
    child_chessboard.curr_color = (child_chessboard.curr_color == Red) ? Black : Red;
    // 判断是否游戏结束
    if (move.is_eat) {
        if (child_chessboard.curr_color == Red && move.eat_chess_type == RedKing
            || child_chessboard.curr_color == Black && move.eat_chess_type == BlackKing) {
            child_chessboard.is_stop_game = true;
        }
    }
    // 移动棋子
    child_chessboard.chessboard_matrix[move.next_x][move.next_y] = child_chessboard.chessboard_matrix[move.init_x][move.init_y];
    child_chessboard.chessboard_matrix[move.init_x][move.init_y] = Empty;
    // 更新所有合法的走法
    child_chessboard.updateMoves();
    return child_chessboard;
}

void ChessBoard::getRookMoves(int x, int y) {
    ChessType chess_type = chessboard_matrix[x][y];  // 当前位置的棋子
    //前后左右分别进行搜索，遇到棋子停止，不同阵营可以吃掉
    for (int i = x + 1; i < 9; i++) {
        Move move(chess_type, x, y, i, y);
        if (chessboard_matrix[i][y] != Empty) {
            ChessColor obstacle_color = getChessColor(chessboard_matrix[i][y]);
            if (obstacle_color != curr_color) {
                // 车可以吃掉对方的棋子
                move.score = getMoveValue(chessboard_matrix[i][y]);
                move.is_eat = true;
                move.eat_chess_type = chessboard_matrix[i][y];
                moves.push_back(move);
            }
            break;
        }
        moves.push_back(move);
    }

    for (int i = x - 1; i >= 0; i--) {
        Move move(chess_type, x, y, i, y);
        if (chessboard_matrix[i][y] != Empty) {
            ChessColor obstacle_color = getChessColor(chessboard_matrix[i][y]);
            if (obstacle_color != curr_color) {
                move.score = getMoveValue(chessboard_matrix[i][y]);
                move.is_eat = true;
                move.eat_chess_type = chessboard_matrix[i][y];
                moves.push_back(move);
            }
            break;
        }
        moves.push_back(move);
    }

    for (int j = y + 1; j < 10; j++) {
        Move move(chess_type, x, y, x, j);
        if (chessboard_matrix[x][j] != Empty) {
            ChessColor obstacle_color = getChessColor(chessboard_matrix[x][j]);
            if (obstacle_color != curr_color) {
                move.score = getMoveValue(chessboard_matrix[x][j]);
                move.is_eat = true;
                move.eat_chess_type = chessboard_matrix[x][j];
                moves.push_back(move);
            }
            break;
        }
        moves.push_back(move);
    }

    for (int j = y - 1; j >= 0; j--) {
        Move move(chess_type, x, y, x, j);
        if (chessboard_matrix[x][j] != Empty) {
            ChessColor obstacle_color = getChessColor(chessboard_matrix[x][j]);
            if (obstacle_color != curr_color) {
                move.score = getMoveValue(chessboard_matrix[x][j]);
                move.is_eat = true;
                move.eat_chess_type = chessboard_matrix[x][j];
                moves.push_back(move);
            }
            break;
        }
        moves.push_back(move);
    }
}

void ChessBoard::getKnightMoves(int x, int y) {
    // 马的走法：走日字，每次走一步，不能被蹩马腿
    ChessType chess_type = chessboard_matrix[x][y];  // 当前位置的棋子
    int dx[] = {-1, 1, 2, 2, 1, -1, -2, -2};
    int dy[] = {2, 2, 1, -1, -2, -2, -1, 1};
    int block_x[] = {0, 0, 1, 1, 0, 0, -1, -1};
    int block_y[] = {1, 1, 0, 0, -1, -1, 0, 0}; // 蹩马腿
    for (int i = 0; i < 8; i++) {
        int next_x = x + dx[i];
        int next_y = y + dy[i];
        // 跳出棋盘，丢弃
        if (next_x < 0 || next_x > 8 || next_y < 0 || next_y > 9) continue;
        // 蹩马腿，丢弃
        if (chessboard_matrix[x + block_x[i]][y + block_y[i]] != Empty) continue;

        Move move(chess_type, x, y, next_x, next_y);
        if (chessboard_matrix[next_x][next_y] != Empty) {
            ChessColor obstacle_color = getChessColor(chessboard_matrix[next_x][next_y]);
            if (obstacle_color != curr_color) {
                // 马可以吃掉对方的棋子
                move.score = getMoveValue(chessboard_matrix[next_x][next_y]);
                move.is_eat = true;
                move.eat_chess_type = chessboard_matrix[next_x][next_y];
                moves.push_back(move);
            }
            // 遇到棋子，如果是己方的棋子，不能走
            continue;
        }
        moves.push_back(move);
    }
}

void ChessBoard::getCannonMoves(int x, int y) {
    // 炮与车的走法类似，但是吃子时需要隔一个子
    // 炮吃子的条件：从炮当前位置的下一个位置开始，搜索到第一个棋子，无论是己方还是对方的棋子，继续搜索，如果遇到第二个棋子，且是对方的棋子，则可以吃掉
    // 炮能走的位置：类似车的位置 + 吃子的位置
    ChessType chess_type = chessboard_matrix[x][y];  // 当前位置的棋子
    for (int i = x + 1; i < 9; i++) {
        Move move(chess_type, x, y, i, y);
        if (chessboard_matrix[i][y] != Empty) {
            // 碰到第一个棋子，无论颜色，继续搜索
            for (int k = i + 1; k < 9; k++) {
                if (chessboard_matrix[k][y] != Empty) {
                    // 碰到第二个棋子，若是对方的棋子，则可以吃掉
                    ChessColor obstacle_color = getChessColor(chessboard_matrix[k][y]);
                    if (obstacle_color != curr_color) {
                        // 炮可以吃掉对方的棋子
                        move.score = getMoveValue(chessboard_matrix[k][y]);
                        move.is_eat = true;
                        move.eat_chess_type = chessboard_matrix[k][y];
                        moves.push_back(move);
                    }
                    break;
                }
            }
            // 无论是否打到对方的棋子，都要跳出外层循环
            break;
        }
        moves.push_back(move);
    }

    for (int i = x - 1; i >= 0; i--) {
        Move move(chess_type, x, y, i, y);
        if (chessboard_matrix[i][y] != Empty) {
            for (int k = i - 1; k >= 0; k--) {
                if (chessboard_matrix[k][y] != Empty) {
                    ChessColor obstacle_color = getChessColor(chessboard_matrix[k][y]);
                    if (obstacle_color != curr_color) {
                        move.score = getMoveValue(chessboard_matrix[k][y]);
                        move.is_eat = true;
                        move.eat_chess_type = chessboard_matrix[k][y];
                        moves.push_back(move);
                    }
                    break;
                }
            }
            break;
        }
        moves.push_back(move);
    }

    for (int j = y + 1; j < 10; j++) {
        Move move(chess_type, x, y, x, j);
        if (chessboard_matrix[x][j] != Empty) {
            for (int k = j + 1; k < 10; k++) {
                if (chessboard_matrix[x][k] != Empty) {
                    ChessColor obstacle_color = getChessColor(chessboard_matrix[x][k]);
                    if (obstacle_color != curr_color) {
                        move.score = getMoveValue(chessboard_matrix[x][k]);
                        move.is_eat = true;
                        move.eat_chess_type = chessboard_matrix[x][k];
                        moves.push_back(move);
                    }
                    break;
                }
            }
            break;
        }
        moves.push_back(move);
    }

    for (int j = y - 1; j >= 0; j--) {
        Move move(chess_type, x, y, x, j);
        if (chessboard_matrix[x][j] != Empty) {
            for (int k = j - 1; k >= 0; k--) {
                if (chessboard_matrix[x][k] != Empty) {
                    ChessColor obstacle_color = getChessColor(chessboard_matrix[x][k]);
                    if (obstacle_color != curr_color) {
                        move.score = getMoveValue(chessboard_matrix[x][k]);
                        move.is_eat = true;
                        move.eat_chess_type = chessboard_matrix[x][k];
                        moves.push_back(move);
                    }
                    break;
                }
            }
            break;
        }
        moves.push_back(move);
    }
}

void ChessBoard::getAdvisorMoves(int x, int y) {
    // 士的走法：走斜线，每次走一步，不能出九宫格
    // 红士的九宫格：3 <= x <= 5, 0 <= y <= 2
    // 黑士的九宫格：3 <= x <= 5, 7 <= y <= 9
    ChessType chess_type = chessboard_matrix[x][y];  // 当前位置的棋子
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

        Move move(chess_type, x, y, next_x, next_y);
        if (chessboard_matrix[next_x][next_y] != Empty) {
            ChessColor obstacle_color = getChessColor(chessboard_matrix[next_x][next_y]);
            if (obstacle_color != curr_color) {
                move.score = getMoveValue(chessboard_matrix[next_x][next_y]);
                move.is_eat = true;
                move.eat_chess_type = chessboard_matrix[next_x][next_y];
                moves.push_back(move);
            }
            // 遇到棋子，如果是己方的棋子，不能走
            continue;
        }
        moves.push_back(move);
    }
}

void ChessBoard::getBishopMoves(int x, int y) {
    // 象的走法：走田字，每次走两步，不能过河，并且不能被塞象眼
    // 红象的范围：0 <= x <= 8, 0 <= y <= 4
    // 黑象的范围：0 <= x <= 8, 5 <= y <= 9
    ChessType chess_type = chessboard_matrix[x][y];  // 当前位置的棋子
    int dx[] = {2, 2, -2, -2};
    int dy[] = {2, -2, 2, -2};
    int block_x[] = {1, 1, -1, -1};
    int block_y[] = {1, -1, 1, -1};
    for (int i = 0; i < 4; i++) {
        int next_x = x + dx[i];
        int next_y = y + dy[i];
        if (curr_color == Red) {
            if (next_x < 0 || next_x > 8 || next_y < 0 || next_y > 4) continue;
        } else {
            if (next_x < 0 || next_x > 8 || next_y < 5 || next_y > 9) continue;
        }

        Move move(chess_type, x, y, next_x, next_y);
        if (chessboard_matrix[next_x][next_y] != Empty) continue;
        if (chessboard_matrix[x + block_x[i]][y + block_y[i]] != Empty) continue;
        if (chessboard_matrix[next_x][next_y] != Empty) {
            ChessColor obstacle_color = getChessColor(chessboard_matrix[next_x][next_y]);
            if (obstacle_color != curr_color) {
                move.score = getMoveValue(chessboard_matrix[next_x][next_y]);
                move.is_eat = true;
                move.eat_chess_type = chessboard_matrix[next_x][next_y];
                moves.push_back(move);
            }
            // 遇到棋子，如果是己方的棋子，不能走
            continue;
        }
        moves.push_back(move);
    }
}

void ChessBoard::getPawnMoves(int x, int y) {
    // 卒的走法：红卒向上走，黑卒向下走，每次走一步，过河后可以左右走
    // 红卒的未过河位置：3 <= y <= 4，只能向 y 增大的方向走
    // 红卒的过河位置：5 <= y <= 9
    // 黑卒的未过河位置：5 <= y <= 6，只能向 y 减小的方向走
    // 黑卒的过河位置：0 <= y <= 4
    ChessType chess_type = chessboard_matrix[x][y];  // 当前位置的棋子
    if (curr_color == Red) {
        // 红卒未过河
        if (y >= 3 && y <= 4) {
            int next_y = y + 1;
            Move move(chess_type, x, y, x, next_y);
            if (chessboard_matrix[x][next_y] != Empty) {
                ChessColor obstacle_color = getChessColor(chessboard_matrix[x][next_y]);
                if (obstacle_color != curr_color) {
                    move.score = getMoveValue(chessboard_matrix[x][next_y]);
                    move.is_eat = true;
                    move.eat_chess_type = chessboard_matrix[x][next_y];
                    moves.push_back(move);
                }
            } else {
                moves.push_back(move);
            }
        } else if (y >= 5 && y <= 9) { // 红卒过河
            int dx[] = {-1, 1, 0};
            int dy[] = {0, 0, 1};
            for (int i = 0; i < 3; i++) {
                int next_x = x + dx[i];
                int next_y = y + dy[i];
                if (next_x < 0 || next_x > 8 || next_y < 5 || next_y > 9) continue;
                Move move(chess_type, x, y, next_x, next_y);
                if (chessboard_matrix[next_x][next_y] != Empty) {
                    ChessColor obstacle_color = getChessColor(chessboard_matrix[next_x][next_y]);
                    if (obstacle_color != curr_color) {
                        move.score = getMoveValue(chessboard_matrix[next_x][next_y]);
                        move.is_eat = true;
                        move.eat_chess_type = chessboard_matrix[next_x][next_y];
                        moves.push_back(move);
                    }
                    continue;
                }
                moves.push_back(move);
            }
        } else {
            std::cout << "Invalid red pawn position" << std::endl;
        }
    } else {
        // 黑卒未过河
        if (y >= 5 && y <= 6) {
            int next_y = y - 1;
            Move move(chess_type, x, y, x, next_y);
            if (chessboard_matrix[x][next_y] != Empty) {
                ChessColor obstacle_color = getChessColor(chessboard_matrix[x][next_y]);
                if (obstacle_color != curr_color) {
                    move.score = getMoveValue(chessboard_matrix[x][next_y]);
                    move.is_eat = true;
                    move.eat_chess_type = chessboard_matrix[x][next_y];
                    moves.push_back(move);
                }
            } else {
                moves.push_back(move);
            }
        } else if (y >= 0 && y <= 4) { // 黑卒过河
            int dx[] = {-1, 1, 0};
            int dy[] = {0, 0, -1};
            for (int i = 0; i < 3; i++) {
                int next_x = x + dx[i];
                int next_y = y + dy[i];
                if (next_x < 0 || next_x > 8 || next_y < 0 || next_y > 4) continue;

                Move move(chess_type, x, y, next_x, next_y);
                if (chessboard_matrix[next_x][next_y] != Empty) {
                    ChessColor obstacle_color = getChessColor(chessboard_matrix[next_x][next_y]);
                    if (obstacle_color != curr_color) {
                        move.score = getMoveValue(chessboard_matrix[next_x][next_y]);
                        move.is_eat = true;
                        move.eat_chess_type = chessboard_matrix[next_x][next_y];
                        moves.push_back(move);
                    }
                    continue;
                }
                moves.push_back(move);
            }
        } else {
            std::cout << "Invalid black pawn position" << std::endl;
        }
    }
}

void ChessBoard::getKingMoves(int x, int y) {
    // 将的走法：走直线，每次走一步，不能出九宫格
    // 红帅的九宫格：3 <= x <= 5, 0 <= y <= 2
    // 黑将的九宫格：3 <= x <= 5, 7 <= y <= 9
    // 将帅不能直接对面，即不能在同一列上，为了表示棋局的结束，认为当将帅对面时，本方可以将对方将/帅吃掉
    ChessType chess_type = chessboard_matrix[x][y];  // 当前位置的棋子
    // 首先考虑将帅直接对面的情况
    if (curr_color == Red) {
        for (int j = y + 1; j < 10; j++) {
            if (chessboard_matrix[x][j] != Empty) {
                if (chessboard_matrix[x][j] == BlackKing) {
                    Move move(chess_type, x, y, x, j);
                    move.score = getMoveValue(chessboard_matrix[x][j]);
                    move.is_eat = true;
                    move.eat_chess_type = chessboard_matrix[x][j];
                    moves.push_back(move);
                }
                break;
            }
        }
    } else {
        for (int j = y - 1; j >= 0; j--) {
            if (chessboard_matrix[x][j] != Empty) {
                if (chessboard_matrix[x][j] == RedKing) {
                    Move move(chess_type, x, y, x, j);
                    move.score = getMoveValue(chessboard_matrix[x][j]);
                    move.is_eat = true;
                    move.eat_chess_type = chessboard_matrix[x][j];
                    moves.push_back(move);
                }
                break;
            }
        }
    }

    int dx[] = {1, 0, -1, 0};
    int dy[] = {0, 1, 0, -1};
    for (int i = 0; i < 4; i++) {
        int next_x = x + dx[i];
        int next_y = y + dy[i];
        if (curr_color == Red) {
            if (next_x < 3 || next_x > 5 || next_y < 0 || next_y > 2) continue;
        } else {
            if (next_x < 3 || next_x > 5 || next_y < 7 || next_y > 9) continue;
        }

        Move move(chess_type, x, y, next_x, next_y);
        if (chessboard_matrix[next_x][next_y] != Empty) {
            ChessColor obstacle_color = getChessColor(chessboard_matrix[next_x][next_y]);
            if (obstacle_color != curr_color) {
                move.score = getMoveValue(chessboard_matrix[next_x][next_y]);
                move.is_eat = true;
                move.eat_chess_type = chessboard_matrix[next_x][next_y];
                moves.push_back(move);
            }
            // 遇到棋子，如果是己方的棋子，不能走
            continue;
        }
        moves.push_back(move);
    }
}




