#include "Astar.h"

Astar::Astar(std::string &input_file, std::string &output_file) {
    this->map = Map(input_file);
    this->output_file = output_file;
}

void Astar::GetResult() {
    /*
     * step_nums = -1 代表无解
     * way 中存储路径，U 代表上，D 代表下，L 代表左，R 代表右
     */
    if (close_list.empty()) {
        step_nums = -1;
        return;
    }
    Point *cur = close_list.back();
    while (cur->getParent() != nullptr) {
        Point *parent = cur->getParent();
        if (parent->getX() == cur->getX() - 1) { way = "U" + way; }
        else if (parent->getX() == cur->getX() + 1) { way = "D" + way; }
        else if (parent->getY() == cur->getY() - 1) { way = "L" + way; }
        else if (parent->getY() == cur->getY() + 1) { way = "R" + way; }
        cur = parent;
    }
}
