#ifndef ASTAR_ASTAR_H
#define ASTAR_ASTAR_H

#include "Point.h"
#include "Map.h"
#include <string>
#include <vector>

class Astar {
private:
    Map map;
    int step_nums;
    std::string way;
    std::string output_file;
    std::vector<Point *> open_list;
    std::vector<Point *> close_list;

public:
    Astar(std::string &input_file, std::string &output_file);
    ~Astar() = default;

    void AstarSearch();

private:
    void HeuristicFunction();
    void GetResult();
    void OutputToFile();
};


#endif //ASTAR_ASTAR_H
