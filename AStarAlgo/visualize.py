import matplotlib.pyplot as plt
import numpy as np
from Map import Map
from typing import List, Tuple


def plot_map(map: Map, path: List[Tuple[int, int]], obstacles: List[Tuple[int, int]]):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0, map.length)
    ax.set_ylim(0, map.width)

    # 绘制网格线
    ax.set_xticks(np.arange(0, map.length + 1, 1))
    ax.set_yticks(np.arange(0, map.width + 1, 1))
    ax.grid(True, which='both', color='black', linestyle='-', linewidth=0.5)

    # 绘制 obstacles 为灰色
    for point in obstacles:
        ax.add_patch(plt.Rectangle((point[0], point[1]), 1, 1, color='gray', edgecolor='black'))

    # 绘制 path 为蓝色，不包括起点和终点
    for point in path[1: -1]:
        ax.add_patch(plt.Rectangle((point[0], point[1]), 1, 1, color='blue', edgecolor='black'))

    # 绘制起点为绿色，终点为红色
    ax.add_patch(plt.Rectangle((map.start.x, map.start.y), 1, 1, color='green', edgecolor='black'))
    ax.add_patch(plt.Rectangle((map.end.x, map.end.y), 1, 1, color='red', edgecolor='black'))

    # 设置边框和背景
    ax.set_facecolor('white')
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor('black')

    # 隐藏主要刻度轴
    ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False,
                   labelleft=False)

    plt.savefig('map_path.png')
    plt.show()


def plot_points(points):
    x, y = zip(*points)
    plt.scatter(x, y)
    plt.plot(x, y, 'ro')
    plt.show()
