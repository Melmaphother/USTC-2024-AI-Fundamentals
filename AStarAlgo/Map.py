from Point import Point
from random import randint
from typing import List, Tuple


class Map:
    def __init__(self, length=50, width=50, is_auto_set_start_end=False, is_randomly_set_obstacles=False):
        self.points = []  # 存储所有点
        self.obstacle_count = 0
        self.obstacles = []  # 存储障碍点
        self.start = self.end = None
        if 5 <= length <= 100 and 5 <= width <= 100:
            self.length = length
            self.width = width
        else:
            raise ValueError("Length and width must be between 5 and 100.")

        self._initialize_points()

        if is_auto_set_start_end:
            self.start = self.points[0][0]
            self.end = self.points[self.length - 1][self.width - 1]
        if is_randomly_set_obstacles:
            self._initialize_obstacles()

    def _initialize_points(self):
        self.points = [[Point(x, y) for y in range(self.width)] for x in range(self.length)]

    def _initialize_obstacles(self):
        self.obstacle_count = (self.length * self.width) // 5
        for _ in range(self.obstacle_count):
            while True:
                x, y = randint(0, self.length - 1), randint(0, self.width - 1)
                if (x, y) != self.start.coordinates() and (x, y) != self.end.coordinates() and not self.points[x][
                    y].is_obstacle:
                    self.points[x][y].set_obstacle()
                    self.obstacles.append((x, y))
                    break

    def set_start_end(self, start_end: Tuple[int, int, int, int]):
        x1, y1, x2, y2 = start_end
        if 0 <= x1 < self.length and 0 <= y1 < self.width and 0 <= x2 < self.length and 0 <= y2 < self.width:
            self.start = self.points[x1][y1]
            self.end = self.points[x2][y2]
            self.start.is_obstacle = False  # 确保起点不是障碍
            self.end.is_obstacle = False  # 确保终点不是障碍
        else:
            raise ValueError("Start and end points must be within map boundaries.")

    def set_horizontal_obstacles(self, n: int, start_point: Tuple[int, int]):
        """设置水平障碍物
        :param n: 障碍物长度
        :param start_point: 障碍物起点，即障碍物最左
        """
        x, y = start_point
        for i in range(n):
            if 0 <= x + i < self.length and 0 <= y < self.width and (x + i, y) != self.start.coordinates() and (
                    x + i, y) != self.end.coordinates():
                if not self.points[x + i][y].is_obstacle:
                    self.points[x + i][y].set_obstacle()
                    self.obstacle_count += 1
                    self.obstacles.append((x + i, y))

            # 碰端点自动截断，碰起点终点自动跳过

    def set_vertical_obstacles(self, n: int, start_point: Tuple[int, int]):
        """设置垂直障碍物
        :param n: 障碍物长度
        :param start_point: 障碍物起点，即障碍物最下
        """
        x, y = start_point
        for i in range(n):
            if 0 <= y + i < self.width and 0 <= x < self.length and (x, y + i) != self.start.coordinates() and (
                    x, y + i) != self.end.coordinates():
                if not self.points[x][y + i].is_obstacle:
                    self.points[x][y + i].set_obstacle()
                    self.obstacle_count += 1
                    self.obstacles.append((x, y + i))

    def set_left_diagonal_obstacles(self, n: int, start_point: Tuple[int, int]):
        """设置左对角线障碍物
        :param n: 障碍物长度
        :param start_point: 障碍物起点，即障碍物左下角
        """
        x, y = start_point
        for i in range(n):
            if 0 <= x + i < self.length and 0 <= y + i < self.width and (x + i, y + i) != self.start.coordinates() and (
                    x + i, y + i) != self.end.coordinates():
                if not self.points[x + i][y + i].is_obstacle:
                    self.points[x + i][y + i].set_obstacle()
                    self.obstacle_count += 1
                    self.obstacles.append((x + i, y + i))

    def set_right_diagonal_obstacles(self, n: int, start_point: Tuple[int, int]):
        """设置右对角线障碍物
        :param n: 障碍物长度
        :param start_point: 障碍物起点，即障碍物左上角
        """
        x, y = start_point
        for i in range(n):
            if 0 <= x + i < self.length and 0 <= y - i < self.width and (x + i, y - i) != self.start.coordinates() and (
                    x + i, y - i) != self.end.coordinates():
                if not self.points[x + i][y - i].is_obstacle:
                    self.points[x + i][y - i].set_obstacle()
                    self.obstacle_count += 1
                    self.obstacles.append((x + i, y - i))

    def remove_obstacles(self, point:Tuple[int, int]):
        x, y = point
        if 0 <= x < self.length and 0 <= y < self.width:
            self.points[x][y].is_obstacle = False
            self.obstacle_count -= 1
            self.obstacles.remove(point)

    def get_neighbors(self, point) -> List[Point]:
        neighbors = []
        x, y = point.coordinates()
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 or j == 0:
                    # 如果在上下左右，只需要考虑其本身是否是障碍
                    if 0 <= x + i < self.length and 0 <= y + j < self.width and (i, j) != (0, 0):
                        if not self.points[x + i][y + j].is_obstacle:
                            neighbors.append(self.points[x + i][y + j])
                else:
                    # 如果在对角线上，需要考虑其本身是否是障碍且需要考虑其是否被斜向障碍挡住
                    if 0 <= x + i < self.length and 0 <= y + j < self.width and (i, j) != (0, 0):
                        if not self.points[x + i][y + j].is_obstacle:
                            if not (self.points[x][y + i].is_obstacle and self.points[x + i][y].is_obstacle):
                                neighbors.append(self.points[x + i][y + j])

        return neighbors
