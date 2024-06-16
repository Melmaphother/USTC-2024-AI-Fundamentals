#include "Astar.h"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <queue>
#include <utility>

Astar::Astar(std::string &input_file, std::string &output_file, std::string _heuristic_function) {
    map = Map(input_file);
    step_nums = 0;

    this->output_file = output_file;

    this->heuristic_type = std::move(_heuristic_function);
    status = AStarInitialization;
}

void Astar::AstarSearch() {
    status = AStarSearching;
    // 起点拥有最大的补给，将起点加入 open_list
    open_list.emplace(map.getStart());
    while (!open_list.empty()) {
        auto curr_point_pos = open_list.begin();
        // 取出并删去 multiset 中的第一个元素
        Point curr_point = *curr_point_pos;
        open_list.erase(curr_point_pos);

        close_list.push_back(curr_point);

        if (curr_point == map.getEnd()) {
            status = AStarFound;
            getResult();
            return;
        }
        std::vector<Point> neighbors = map.getNeighbors(curr_point);
        for (auto &neighbor: neighbors) {
            auto close_list_iter = std::find_if(close_list.begin(), close_list.end(), [neighbor](const Point &p) {
                return p == neighbor;
            });
            auto open_list_iter = std::find_if(open_list.begin(), open_list.end(), [neighbor](const Point &p) {
                return p == neighbor;
            });
            bool is_in_close_list = close_list_iter != close_list.end();
            bool is_in_open_list = open_list_iter != open_list.end();

            // int neighbor_g = neighbor.distance(map.getStart());
            int neighbor_g = curr_point.g + 1; // 两个都行其实
            // 非补给点的补给量为当前点的补给量减 1，补给点的补给量为其最大补给量（也就是本身）
            int neighbor_supply = neighbor.type == PointType::Supply ? neighbor.supply : curr_point.supply - 1;
            int neighbor_h = heuristicFunc(neighbor, neighbor_supply);

            if (heuristic_type == "non-trivial" && neighbor_h == -1) {
                continue;
            }
            if (!is_in_close_list && !is_in_open_list) {
                neighbor.supply = neighbor_supply;
                neighbor.g = neighbor_g;
                neighbor.h = neighbor_h;
                neighbor.parent = curr_point.getPos();
                open_list.emplace(neighbor);
            } else if (is_in_open_list) {
                // 如果 neighbor_g < open list 中该点的 g 值，则更新 open_list中该点 g 值
                Point neighbor_in_open_list = *open_list_iter;
                if (neighbor_g < neighbor_in_open_list.g) {
                    neighbor_in_open_list.g = neighbor_g;
                    neighbor_in_open_list.h = neighbor_h;
                    neighbor_in_open_list.parent = curr_point.getPos();
                    open_list.erase(open_list_iter);
                    open_list.emplace(neighbor_in_open_list);
                }
            } else {
                // 即使在 close_list 中，若新的结点的 supply 更多，也要重新加入 open_list
                Point neighbor_in_close_list = *close_list_iter;
                if (neighbor_supply > neighbor_in_close_list.supply) {
                    neighbor_in_close_list.supply = neighbor_supply;
                    neighbor_in_close_list.g = neighbor_g;
                    neighbor_in_close_list.h = neighbor_h;
                    neighbor_in_close_list.parent = curr_point.getPos();
                    open_list.emplace(neighbor_in_close_list);
                }
            }
        }
    }
    status = AStarNotFound;
    getResult();
}

/**
 * @brief 启发式函数
 */
int Astar::heuristicFunc(Point &point, int curr_supply) {
    if (heuristic_type == "trivial") {
        if (curr_supply <= 0) {
            return -1;
        }
        return point.distance(map.getEnd());
    } else if (heuristic_type == "non-trivial") {
        /**
        * 考虑到只能上下左右移动，在当前拥有的补给为 r 时，可到的范围是以当前点为中心的一个正方形（旋转 45 度），其对角线长度为 2r + 1
        * 设这个范围为 SupplyRegion，启发式函数设计如下：
        * 1. 若终点在 SupplyRegion 中，则启发式函数为终点到当前点的曼哈顿距离
        * 2. 若终点不在 SupplyRegion 中，则遍历所有补给点，将在 SupplyRegion 中的所有补给点到当前点的曼哈顿距离加上终点到该补给点的曼哈顿距离
        * 进小根堆，取最小值作为启发式函数
        * 3. 若在 2 中找不到任何一个补给点，则返回 -1 告知算法需要重新规划路径
        */
        if (isInSupplyRegion(map.getEnd().getPos(), point.getPos(), curr_supply)) {
            return point.distance(map.getEnd());
        } else {
            std::priority_queue<int, std::vector<int>, std::greater<>> distances;  // 从小到大排序
            for (auto &supply_point: map.getSupplyPoints()) {
                if (isInSupplyRegion(supply_point, point.getPos(), curr_supply)) {
                    distances.push(point.distance(Point(supply_point)) + Point(supply_point).distance(map.getEnd()));
                }
            }
            if (distances.empty()) {
                return -1;
            }
            return distances.top();
        }
    } else {
        std::cout << R"(Error: heuristic type is not "trivial" or "non-trivial"!)" << std::endl;
        return -1;
    }
}

/**
 * @brief 判断点是否在当前补给可达最大范围内
 */
inline bool Astar::isInSupplyRegion(std::pair<int, int> point_pos, std::pair<int, int> center_point_pos, int r) {
    int x = point_pos.first;
    int y = point_pos.second;
    int center_x = center_point_pos.first;
    int center_y = center_point_pos.second;
    return abs(x - center_x) + abs(y - center_y) <= r;
}


/**
 * @brief 反向遍历 close_list，获取路径
 */
void Astar::getResult() {
    /*
     * step_nums = -1 代表无解
     * way 中存储路径，U 代表上，D 代表下，L 代表左，R 代表右
     */
    if (status == AStarNotFound) {
        step_nums = -1;
    } else if (status == AStarFound) {
        // 反向遍历 close_list，获取路径
        std::reverse(close_list.begin(), close_list.end());
        auto it = close_list.begin();
        std::pair<int, int> curr_pos = it->getPos();
        std::pair<int, int> parent_pos = it->parent;
        for (it = close_list.begin(); it != close_list.end(); it++) {
            auto pos = it->getPos();
            if (pos == parent_pos) {
                if (curr_pos.first == parent_pos.first) {
                    if (curr_pos.second < parent_pos.second) {
                        way = "D" + way;
                    } else {
                        way = "U" + way;
                    }
                } else {
                    if (curr_pos.first < parent_pos.first) {
                        way = "L" + way;
                    } else {
                        way = "R" + way;
                    }
                }
                curr_pos = pos;
                parent_pos = it->parent;
            }
        }
        step_nums = static_cast<int>(way.size());
    } else {
        std::cout << "Error: status is not AStarFound or AStarNotFound!" << std::endl;
    }
    outputToFile();
}

/**
 * @brief 将结果输出到文件
 */
void Astar::outputToFile() {
    std::ofstream file(output_file);
    if (!file.is_open()) {
        std::cout << "Error opening file!" << std::endl;
        return;
    }
    if (status == AStarNotFound) {
        std::cout << "No way to go!" << std::endl;
        file << step_nums << std::endl;
    } else if (status == AStarFound) {
        std::cout << "Find the way!" << std::endl;
        std::cout << "All cells: " << map.getMapSize() << std::endl;
        std::cout << "Searched cells: " << close_list.size() << std::endl;
        std::cout << "Step nums: " << step_nums << std::endl;
        file << step_nums << std::endl;
        file << way << std::endl;
    } else {
        std::cout << "Error: status is not AStarFound or AStarNotFound!" << std::endl;
    }
    file.close();
}
