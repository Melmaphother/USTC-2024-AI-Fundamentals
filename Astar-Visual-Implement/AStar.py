import heapq


class PriorityQueue:
    def __init__(self):
        self.elements = []
        heapq.heapify(self.elements)

    def empty(self):
        return not self.elements

    def put(self, item, priority):
        # 如果优先级相同，heapq会尝试比较元组中的下一个元素。
        # 所以 Point 中也需要实现 __lt__ 方法
        heapq.heappush(self.elements, (priority, item))

    def get(self):
        return heapq.heappop(self.elements)[1]

    def is_include(self, item):
        return item in [e[1] for e in self.elements]

    def __len__(self):
        return len(self.elements)


class AStar:
    def __init__(self, _map):
        self.map = _map
        self.open_list = PriorityQueue()
        self.close_list = []

    def find_path(self):
        self.open_list.put(self.map.start, 0.0)

        while len(self.open_list) > 0:
            curr_point = self.open_list.get()
            if curr_point == self.map.end:
                return self._get_path(curr_point)

            self.close_list.append(curr_point)
            neighbors = self.map.get_neighbors(curr_point)
            for neighbor in neighbors:
                if neighbor in self.close_list:
                    continue
                G = curr_point.G + curr_point.distance(neighbor)  # 计算从起点到当前点再到邻居点的G值计算G值
                if self.open_list.is_include(neighbor):
                    # 如果邻居点在open_list中，说明已经计算过G值，这里需要判断新的G值是否更小
                    if G < neighbor.G:
                        # 如果新的G值比邻居当前的G值小，说明找到了更好的路径如果G值小于邻居点的G值，则更新邻居点的G值和parent
                        neighbor.G = G
                        neighbor.parent = curr_point
                else:
                    # 如果邻居点不在open_list中，说明还没有计算过G值，这里需要计算G值并加入open_list
                    neighbor.G = G
                    neighbor.H = neighbor.distance(self.map.end)
                    neighbor.parent = curr_point
                    self.open_list.put(neighbor, neighbor.get_F())

    def _get_path(self, point):
        path = []
        while point != self.map.start:
            path.append(point.coordinates())
            point = point.parent
        path.append(self.map.start.coordinates())
        return path[::-1]  # 逆序返回路径
