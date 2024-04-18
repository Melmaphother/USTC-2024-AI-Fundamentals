#ifndef ASTAR_ASTAR_H
#define ASTAR_ASTAR_H

#include "Point.h"
#include "Map.h"
#include <string>
#include <vector>
#include <queue>
typedef std::pair<Point *, int> SearchPoint; // 点以及该点上拥有的补给

struct CompareF {
    bool operator()(const SearchPoint &p1, const SearchPoint &p2) {
        return p1.first->getF() > p2.first->getF();  // 小根堆
    }
};

class Astar {
private:
    Map map;
    Point *start;
    Point *end;
    int max_supply;

    int step_nums;
    std::string way;
    std::vector<SearchPoint> open_list;
    std::vector<SearchPoint> close_list;

    std::string output_file;

public:
    Astar(std::string &input_file, std::string &output_file);

    ~Astar() = default;

    void AstarSearch();

private:
    int HeuristicFunction(Point *point, int curr_supply);  // 启发式函数
    static bool isInSupplyRegion(Point *point, Point *center_point, int r);  // 判断点是否在当前补给可达最大范围内
    void GetResult();  // 反向遍历 close_list，获取路径
    void OutputToFile();  // 将结果输出到文件
};


#endif //ASTAR_ASTAR_H
