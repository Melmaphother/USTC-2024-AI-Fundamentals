#include "Point.h"
#include "Map.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <utility>
#include <algorithm>

/**
 * @brief 构造函数，从文件中读取地图信息，并对所有补给点进行预处理
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
    this->length = std::stoi(words[0]);
    this->width = std::stoi(words[1]);
    this->supply = std::stoi(words[2]);

    map = new Point *[length];
    // 加载地图
    for (int i = 0; i < length; i++) {
        map[i] = new Point[width];
        getline(file, line);
        std::stringstream ss(line);
        for (int j = 0; j < width; j++) {
            ss >> word;
            int type = std::stoi(word);
            map[i][j].setType(type);
            map[i][j].setX(i);
            map[i][j].setY(j);
            if (type == 3) {
                start = &map[i][j];
            } else if (type == 4) {
                end = &map[i][j];
            } else if (type == 2) {
                supply_points.push_back(&map[i][j]);
            }
        }
    }
    file.close();

    // 预处理部分
    PreProcessSupplyPoints();
}

Map::~Map() {
    for (int i = 0; i < length; i++) {
        delete[] map[i];
    }
    delete[] map;
}

/**
 * @brief 获取周围所有非障碍点
 * @param point 当前点
 * @return 周围所有非障碍点
 */
std::vector<Point *> Map::getNeighbors(Point *point) {
    int x = point->getX();
    int y = point->getY();
    std::vector<Point *> neighbors;
    std::vector<std::pair<int, int>> directions = {{0,  1},
                                                   {0,  -1},
                                                   {1,  0},
                                                   {-1, 0}};
    for (auto direction: directions) {
        int new_x = x + direction.first;
        int new_y = y + direction.second;
        if (isInMap(&map[new_x][new_y]) && map[new_x][new_y].getType() != 1) {
            neighbors.push_back(&map[new_x][new_y]);
        }
    }
    return neighbors;
}

/**
 * @brief 判断点是否在地图内
 * @param point 待判断点
 * @return 在地图内返回true，否则返回false
 */
inline bool Map::isInMap(Point *point) const {
    int x = point->getX();
    int y = point->getY();
    return x >= 0 && x < length && y >= 0 && y < width;
}

/**
 * @brief 预处理所有补给点到终点的曼哈顿距离，并依照距离对所有补给点从小到大排序
 */
void Map::PreProcessSupplyPoints() {
    std::sort(supply_points.begin(), supply_points.end(), [this](Point *a, Point *b) {
        return a->distance(end) < b->distance(end);
    });
}
