
#ifndef ASTAR_POINT_H
#define ASTAR_POINT_H

#include <cmath>
#include <utility>

enum class PointType {
    Passage = '0',  // 可通行点
    Block = '1',  // 障碍
    Supply = '2',  // 补给点
    Start = '3',  // 起点
    End = '4'  // 终点
};

struct Point {
    int x{};
    int y{};
    std::pair<int, int> parent{};
    int g{};
    int h{};
    int f{};
    PointType type{PointType::Passage};
    int supply{};

    Point() = default;

    explicit Point(std::pair<int, int> point) : x(point.first), y(point.second) {}

    inline bool operator==(const Point &p) const { return x == p.x && y == p.y; }

    inline bool operator!=(const Point &p) const { return x != p.x || y != p.y; }

    inline std::pair<int, int> getPos() const { return {x, y}; }

    inline int distance(const Point &p) const { return abs(x - p.x) + abs(y - p.y); }
};

#endif // ASTAR_POINT_H