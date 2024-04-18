
#ifndef ASTAR_POINT_H
#define ASTAR_POINT_H

#include <cmath>

class Point {
private:
    int x;
    int y;
    Point *parent;
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
        x = y = g = h = f = type = 0;
        parent = nullptr;
    }

    inline bool operator<(const Point &p) const { return f > p.f; }

    inline void setX(int _x) { this->x = _x; }

    inline int getX() const { return x; }

    inline void setY(int _y) { this->y = _y; }

    inline int getY() const { return y; }

    inline void setParent(Point *_parent) { this->parent = _parent; }

    inline Point *getParent() { return parent; }

    inline void setG(int _g) { this->g = _g; }

    inline int getG() const { return g; }

    inline void setH(int _h) { this->h = _h; }

    inline int getH() const { return h; }

    inline int getF() {
        f = g + h;
        return f;
    }

    inline void setType(int _type) { this->type = _type; }

    inline int getType() const { return type; }

    inline bool operator==(const Point *p) const { return x == p->x && y == p->y; }

    inline bool operator!=(const Point *p) const { return x != p->x || y != p->y; }

    inline int distance(const Point *p) const { return abs(x - p->x) + abs(y - p->y); }
};

#endif // ASTAR_POINT_H