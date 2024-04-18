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
    Point** map;
    Point* start;
    Point* end;
    std::vector<Point *> supply_points;  // 所有补给点

public:
    Map(const std::string &input_file);

    Map();

    ~Map();

public:
    // 获取周围所有非障碍点
    std::vector<Point*> getNeighbors(Point* point);

private:
    // 判断点是否在地图内
    bool isInMap(Point* point) const;

    // 预处理所有补给点到终点的曼哈顿距离，并从小到大排序
    void PreProcessSupplyPoints();

};

#endif //ASTAR_MAP_H