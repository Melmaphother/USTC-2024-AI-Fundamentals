#include "Astar.h"
#include <fstream>
#include <iostream>
#include <algorithm>

Astar::Astar(std::string &input_file, std::string &output_file) {
    map = Map(input_file);
    start = Point(map.getStart());
    end = Point(map.getEnd());
    step_nums = 0;
    std::make_heap(open_list.begin(), open_list.end(), CompareF());

    output_file = output_file;
}

void Astar::AstarSearch() {
    // 起点拥有最大的补给，将起点加入 open_list
    open_list.emplace_back(start, map.getSupply());
    std::push_heap(open_list.begin(), open_list.end(), CompareF());
    while (!open_list.empty()) {
        std::pop_heap(open_list.begin(), open_list.end(), CompareF()); // 将 open_list 中的最小值放到最后
        auto curr_point = open_list.back();  // 获取最小值
        open_list.pop_back();  // 删除最后一个元素
        if (curr_point.first == end) {
            GetResult();
        }

        close_list.push_back(curr_point);
        auto neighbors = map.getNeighbors(curr_point.first);
        for (auto &neighbor: neighbors) {
            // 如果邻居点在 close_list 中，跳过
            if (std::find_if(close_list.begin(), close_list.end(), [neighbor](const SearchPoint &p) {
                return p.first == neighbor;
            }) != close_list.end()) {
                continue;
            }
            int neighbor_g = curr_point.first.getG() + 1;
            int neighbor_supply = neighbor.getType() == 2 ? map.getSupply() : curr_point.second - 1;
            // 如果邻居点在open_list中，说明已经计算过G值，这里需要判断新的G值是否更小
            if(std::find_if(open_list.begin(), open_list.end(), [neighbor](const SearchPoint &p) {
                return p.first == neighbor;
            }) != open_list.end()) {
                if (neighbor_g < neighbor.getG()) {
                    neighbor.setG(neighbor_g);
                    neighbor.setF();
                    neighbor.setParent(&curr_point.first);
                }
            }
            else {
                neighbor.setG(neighbor_g);
                neighbor.setF();
                int neighbor_h = HeuristicFunction(neighbor, neighbor_supply);
                if (neighbor_h == -1) {
                    continue;  // 不将neighbor加入open_list
                }
                neighbor.setH(neighbor_h);
                neighbor.setF();
                neighbor.setParent(&curr_point.first);
                open_list.emplace_back(neighbor, neighbor_supply);
                std::push_heap(open_list.begin(), open_list.end(), CompareF());
            }
        }
    }

}

/**
 * @brief 启发式函数
 */
int Astar::HeuristicFunction(Point& point, int curr_supply) {
    /**
     * 考虑到只能上下左右移动，在当前拥有的补给为 r 时，可到的范围是以当前点为中心的一个正方形（旋转 45 度），其对角线长度为 2r + 1
     * 设这个范围为 SupplyRegion，启发式函数设计如下：
     * 1. 若终点在 SupplyRegion 中，则启发式函数为终点到当前点的曼哈顿距离
     * 2. 若终点不在 SupplyRegion 中，则遍历所有补给点，将在 SupplyRegion 中的所有补给点到当前点的曼哈顿距离加上终点到该补给点的曼哈顿距离
     * 进小根堆，取最小值作为启发式函数
     * 3. 若在 2 中找不到任何一个补给点，则返回 -1 告知算法需要重新规划路径
     */
    if (isInSupplyRegion(end, point, curr_supply)) {
        return point.distance(end);
    } else {
        std::priority_queue<int, std::vector<int>, std::greater<>> distances;  // 从小到大排序
        for (auto &supply_point: map.getSupplyPoints()) {
            if (isInSupplyRegion(supply_point, point, curr_supply)) {
                distances.push(supply_point->distance(point) + supply_point->distance(end));
            }
        }
        if (distances.empty()) {
            return -1;
        }
        return distances.top();
    }
}

/**
 * @brief 判断点是否在当前补给可达最大范围内
 */
inline bool Astar::isInSupplyRegion(Point &point, Point &center_point, int r) {
    int x = point.getX();
    int y = point.getY();
    int center_x = center_point.getX();
    int center_y = center_point.getY();
    return abs(x - center_x) + abs(y - center_y) <= r;
}


/**
 * @brief 反向遍历 close_list，获取路径
 */
void Astar::GetResult() {
    /*
     * step_nums = -1 代表无解
     * way 中存储路径，U 代表上，D 代表下，L 代表左，R 代表右
     */
    if (close_list.empty()) {
        step_nums = -1;
        return;
    }
    Point *cur = close_list.back().first;
    while (cur->getParent() != nullptr) {
        Point *parent = cur->getParent();
        if (parent->getX() == cur->getX() - 1) { way = "U" + way; }
        else if (parent->getX() == cur->getX() + 1) { way = "D" + way; }
        else if (parent->getY() == cur->getY() - 1) { way = "L" + way; }
        else if (parent->getY() == cur->getY() + 1) { way = "R" + way; }
        cur = parent;
        step_nums++;
    }
    OutputToFile();
}

/**
 * @brief 将结果输出到文件
 */
void Astar::OutputToFile() {
    std::ofstream file(output_file);
    if (!file.is_open()) {
        std::cout << "Error opening file!" << std::endl;
        return;
    }
    if (step_nums == -1) {
        file << step_nums << std::endl;
    } else {
        file << step_nums << std::endl;
        file << way << std::endl;
    }
    file.close();
}
