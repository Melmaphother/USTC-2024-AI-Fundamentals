#include "Point.h"
#include "Map.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <utility>
#include <cassert>


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
    std::stringstream s1(line);
    std::string word;
    std::vector<std::string> first_line;
    while (s1 >> word) {
        first_line.push_back(word);
    }
    this->height = std::stoi(first_line[0]);
    this->width = std::stoi(first_line[1]);
    this->supply = std::stoi(first_line[2]);

    map_matrix.resize(width);
    for (int i = 0; i < width; i++) {
        map_matrix[i].resize(height);
    }

    for (int j = height - 1; j >= 0; j--) {
        getline(file, line);
        std::stringstream s2(line);
        for (int i = 0; i < width; i++) {
            s2 >> word;
            // 如果 word 不在 ['0', '1', '2', '3', '4'] 中，抛出异常
            assert(word.size() == 1);
            assert(word[0] >= '0' && word[0] <= '4');
            auto type = static_cast<PointType>(word[0]);
            map_matrix[i][j].type = type;
            map_matrix[i][j].x = i;
            map_matrix[i][j].y = j;
            if (type == PointType::Start) {
                // 起点拥有最大补给
                map_matrix[i][j].supply = supply;
                start = map_matrix[i][j];
            } else if (type == PointType::End) {
                end = map_matrix[i][j];
            } else if (type == PointType::Supply) {
                // 补给点拥有最大补给
                map_matrix[i][j].supply = supply;
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
std::vector<Point> Map::getNeighbors(Point &point) {
    if (point.h == -1) {
        // 如果启发式函数值为 -1，说明该点已经饿死，没有邻居
        return {};
    }
    int x = point.x;
    int y = point.y;
    std::vector<Point> neighbors;
    std::vector<std::pair<int, int>> directions = {{0,  1},
                                                   {0,  -1},
                                                   {1,  0},
                                                   {-1, 0}};
    for (auto direction: directions) {
        int new_x = x + direction.first;
        int new_y = y + direction.second;
        if (isInMap(new_x, new_y) && map_matrix[new_x][new_y].type != PointType::Block) {
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

