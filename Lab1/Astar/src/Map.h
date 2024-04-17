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

public:
    Map(const std::string &input_file);
    ~Map();

public:
    // 获取周围四个非障碍点
    std::vector<Point*> getNeighbors(Point* point);
};