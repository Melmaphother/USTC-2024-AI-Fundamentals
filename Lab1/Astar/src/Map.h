#ifndef ASTAR_MAP_H
#define ASTAR_MAP_H

#include "Point.h"
#include <string>
#include <vector>

class Map {
private:
    int length;
    int width;
    int supply;
    Point **map;
    Point *start;
    Point *end;
    std::vector<Point *> supply_points;  // 所有补给点

public:
    Map() = default;

    explicit Map(const std::string &input_file);

    ~Map();

public:
    // 获取周围所有非障碍点
    std::vector<Point *> getNeighbors(Point *point);

    Point *getStart() const { return start; }

    Point *getEnd() const { return end; }

    int getSupply() const { return supply; }

    std::vector<Point *> getSupplyPoints() const { return supply_points; }

private:
    // 判断点是否在地图内
    bool isInMap(Point *point) const;

};

#endif //ASTAR_MAP_H