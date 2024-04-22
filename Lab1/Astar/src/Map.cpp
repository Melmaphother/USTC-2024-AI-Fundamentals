#include "Point.h"
#include "Map.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <utility>


/**
 * @brief 构造函数，从文件中读取地图信息
 * @param input_file 地图文件路径
 */
Map::Map(const std::string &input_file) {
    // 文件读取，获取地图信息
    std::ifstream file(input_file);
    if (!file.is_open()) {
        std::cout << "Error opening file!" << std::endl;
        return;
    }

    std::string line;
    getline(file, line); // 读取第一行
    std::stringstream ss(line);
    std::string word;
    std::vector<std::string> words;
    while (ss >> word) { words.push_back(word); }
    this->height = std::stoi(words[0]);  // 高度 M，y 轴
    this->width = std::stoi(words[1]);  // 宽度 N，x 轴
    this->supply = std::stoi(words[2]);  // 补给 T

    map_matrix.resize(width);
    for (int i = 0; i < width; i++) {
        map_matrix[i].resize(height);
    }

    for (int j = height - 1; j >= 0; j--) {
        getline(file, line);
        std::stringstream ss_2(line);
        for (int i = 0; i < width; i++) {
            ss_2 >> word;
            int type = std::stoi(word);
            map_matrix[i][j].setType(type);
            map_matrix[i][j].setX(i);
            map_matrix[i][j].setY(j);
            map_matrix[i][j].setSupply(0);
            if (type == 3) {
                map_matrix[i][j].setSupply(supply); // 起点拥有最大补给
                start = map_matrix[i][j];
            } else if (type == 4) {
                end = map_matrix[i][j];
            } else if (type == 2) {
                map_matrix[i][j].setSupply(supply);  // 补给点拥有最大补给
                supply_points.emplace_back(i, j);
            }
        }
    }
    file.close();
}


/**
 * @brief 获取周围所有非障碍点
 * @param point 当前点
 * @return 周围所有非障碍点
 */
std::vector<Point> Map::getNeighbors(Point& point) {
    if (point.getH() == -1) {
        return {};  // 如果启发式函数值为 -1，说明该点已经饿死，没有邻居
    }
    int x = point.getX();
    int y = point.getY();
    std::vector<Point> neighbors;
    std::vector<std::pair<int, int>> directions = {{0,  1},
                                                   {0,  -1},
                                                   {1,  0},
                                                   {-1, 0}};
    for (auto direction: directions) {
        int new_x = x + direction.first;
        int new_y = y + direction.second;
        if (isInMap(new_x, new_y) && map_matrix[new_x][new_y].getType() != 1) {
            neighbors.push_back(map_matrix[new_x][new_y]);
        }
    }
    return neighbors;
}

/**
 * @brief 判断点是否在地图内
 * @param x 坐标，y 坐标
 * @return 在地图内返回true，否则返回false
 */
inline bool Map::isInMap(int x, int y) const {
    return x >= 0 && x < width && y >= 0 && y < height;
}

