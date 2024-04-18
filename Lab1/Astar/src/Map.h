#ifndef ASTAR_MAP_H
#define ASTAR_MAP_H

#include "Point.h"
#include <string>
#include <vector>
#include <utility>

class Map {
private:
    int height{};
    int width{};
    int supply{};
    std::vector<std::vector<Point>> map;
    std::pair<int, int> start{};
    std::pair<int, int> end{};
    std::vector<std::pair<int, int>> supply_points;  // 所有补给点

public:
    Map() = default;

    explicit Map(const std::string &input_file);

    ~Map() = default;

public:
    // 获取周围所有非障碍点
    std::vector<Point> getNeighbors(Point &point);

    // 获取地图信息
    int getMapSize() const { return width * height; }

    int getSupply() const { return supply; }

    std::pair<int, int> getStart() const { return start; }

    std::pair<int, int> getEnd() const { return end; }

    std::vector<std::pair<int, int>> getSupplyPoints() const { return supply_points; }

    // 用于回溯路径
    void setParentToMap(std::pair<int, int> point_pos, std::pair<int, int> parent_pos) {
        map[point_pos.first][point_pos.second].setParentPos(parent_pos);
    }

    std::pair<int, int>
    getParentFromMap(std::pair<int, int> point_pos) const { return map[point_pos.first][point_pos.second].getParentPos(); }

private:
    // 判断点是否在地图内
    bool isInMap(int x, int y) const;

};

#endif //ASTAR_MAP_H