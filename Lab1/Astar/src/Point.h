
#ifndef POINT_H
#define POINT_H
#include <cmath>

class Point {
private:
    int x;
    int y;
    Point* parent;
    int g;
    int h;
    int f;

    int type;
    /*
     * 0 ：可通行
       1 ：不可通行
       2 ：补给点
       3 ：起点
       4 ：终点
     */

public:
    Point() {
        x = 0;
        y = 0;
        parent = nullptr;
        g = 0;
        h = 0;
        f = 0;
    }

    bool operator<(const Point& p) const {
        return f > p.f;
    }

    void setX(int _x) {
        this->x = _x;
    }

    void setY(int _y) {
        this->y = _y;
    }

    void setParent(Point* _parent) {
        this->parent = _parent;
    }

    void setG(int _g) {
        this->g = _g;
    }

    void setH(int _h) {
        this->h = _h;
    }

    void getF() {
        f = g + h;
    }

    void setType(int _type) {
        this->type = _type;
    }

    bool operator==(const Point& p) const {
        return x == p.x && y == p.y;
    }

    bool operator!=(const Point& p) const {
        return x != p.x || y != p.y;
    }

    int distance(const Point& p) {
        return abs(x - p.x) + abs(y - p.y);
    }
};

#endif