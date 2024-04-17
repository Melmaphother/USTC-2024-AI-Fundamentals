#include "Point.h"
#include "Map.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

Map::Map(const std::string &input_file) {
    std::ifstream file(input_file);
    if (!file.is_open()) {
        std::cout << "Error opening file!" << std::endl;
        return;
    }

    std::string line;
    getline(file, line); // 读取第一行
    std::stringstream ss(line);
    std::string word;
    std::vector<std::string> words;
    while (ss >> word) { words.push_back(word); }
    this->length = std::stoi(words[0]);
    this->width = std::stoi(words[1]);
    this->supply = std::stoi(words[2]);

    map = new Point*[length];
    // 加载地图
    for (int i = 0; i < length; i++) {
        map[i] = new Point[width];
        getline(file, line);
        std::stringstream ss(line);
        for (int j = 0; j < width; j++) {
            ss >> word;
            int type = std::stoi(word);
            map[i][j].setType(type);
            map[i][j].setX(i);
            map[i][j].setY(j);
            if (type == 3) {
                start = &map[i][j];
            } else if (type == 4) {
                end = &map[i][j];
            }
        }
    }
    file.close();
}

Map::~Map() {
    for (int i = 0; i < length; i++) {
        delete[] map[i];
    }
    delete[] map;
}