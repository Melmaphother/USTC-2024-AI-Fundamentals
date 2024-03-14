class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.is_obstacle = False
        self.G = 0
        self.H = 0
        self.F = 0

    def set_obstacle(self):
        self.is_obstacle = True

    def get_F(self):
        self.F = self.G + self.H
        return self.F

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        # 首先比较 H 值，然后比较 G 值，最后比较坐标
        return self.H < other.H or self.G < other.G or self.x < other.x or self.y < other.y

    def coordinates(self):
        return self.x, self.y

    def _euclidean_distance(self, other_point):
        # 使用欧几里得距离公式计算两点之间的距离
        return ((self.x - other_point.x) ** 2 + (self.y - other_point.y) ** 2) ** 0.5

    def _manhattan_distance(self, other_point):
        # 使用曼哈顿距离公式计算两点之间的距离
        return abs(self.x - other_point.x) + abs(self.y - other_point.y)

    def _diagonal_distance(self, other_point):
        dx = abs(self.x - other_point.x)
        dy = abs(self.y - other_point.y)
        D = 1  # 沿直线移动的成本
        D2 = 1.414  # 沿对角线移动的成本，根据实际情况，这里假设对角线移动的成本为sqrt(2)约等于1.414
        return D * (dx + dy) + (D2 - 2 * D) * min(dx, dy)

    def distance(self, other_point=None, method='diagonal'):
        if method == 'euclidean':
            return self._euclidean_distance(other_point)
        elif method == 'manhattan':
            return self._manhattan_distance(other_point)
        elif method == 'diagonal':
            return self._diagonal_distance(other_point)
        else:
            raise ValueError("Invalid distance calculation method.")
