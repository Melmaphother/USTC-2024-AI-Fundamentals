#ifndef ASTAR_ASTAR_H
#define ASTAR_ASTAR_H

#include "Point.h"
#include "Map.h"
#include <string>
#include <vector>
#include <set>


struct ComparePoint {
    bool operator()(const Point &p1, const Point &p2) {
        return p1.f < p2.f;
    }
};

enum AstarStatus {
    AStarInitialization,
    AStarSearching,
    AStarFound,
    AStarNotFound
};


class Astar {
private:
    Map map;

    int step_nums;
    std::string way;
    std::multiset<Point, ComparePoint> open_list;
    std::vector<Point> close_list;

    std::string output_file;

    std::string heuristic_type;

    int status;

public:
    Astar(std::string &input_file, std::string &output_file, std::string _heuristic_function="trivial");

    ~Astar() = default;

    void AstarSearch();


private:
    int heuristicFunc(Point &point, int curr_supply);  // 启发式函数
    static bool
    isInSupplyRegion(std::pair<int, int> point_pos, std::pair<int, int> center_point_pos, int r);  // 判断点是否在当前补给可达最大范围内
    void getResult();  // 反向遍历 close_list，获取路径
    void outputToFile();  // 将结果输出到文件
};


#endif //ASTAR_ASTAR_H
