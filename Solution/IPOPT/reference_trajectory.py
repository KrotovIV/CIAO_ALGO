import numpy as np
from typing import Tuple, List, Optional
import matplotlib.pyplot as plt
import heapq
from collections import deque


class Node:
    """Узел для алгоритма A*"""

    def __init__(self, x, y, cost, parent=None):
        self.x = x
        self.y = y
        self.cost = cost
        self.parent = parent

    def __lt__(self, other):
        return self.cost < other.cost


def heuristic(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Эвристическая функция (Евклидово расстояние)"""
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)


def a_star_search(occupancy_grid: np.ndarray,
                  start: Tuple[float, float],
                  goal: Tuple[float, float],
                  grid_resolution: float,
                  grid_origin: Tuple[float, float] = (0.0, 0.0)) -> List[Tuple[float, float]]:
    """
    Реализация алгоритма A* для поиска пути на Occupancy Grid.

    :param occupancy_grid: матрица Occupancy Grid (0 - свободно, 1 - занято)
    :param start: начальная точка в мировых координатах (x, y)
    :param goal: целевая точка в мировых координатах (x, y)
    :param grid_resolution: разрешение сетки (метров на ячейку)
    :param grid_origin: начало координат сетки

    :return: список точек пути в мировых координатах
    """
    x0, y0 = grid_origin
    height, width = occupancy_grid.shape

    # Более аккуратное преобразование с проверкой границ
    def world_to_grid(x, y):
        grid_x = int(round((x - x0) / grid_resolution))
        grid_y = int(round((y - y0) / grid_resolution))
        # Ограничиваем индексы границами сетки
        grid_x = max(0, min(width - 1, grid_x))
        grid_y = max(0, min(height - 1, grid_y))
        return grid_x, grid_y

    start_grid = world_to_grid(start[0], start[1])
    goal_grid = world_to_grid(goal[0], goal[1])

    # Проверяем, что старт и цель не в препятствиях
    if occupancy_grid[start_grid[1], start_grid[0]] == 1:
        raise ValueError("Start point is inside an obstacle")
    if occupancy_grid[goal_grid[1], goal_grid[0]] == 1:
        raise ValueError("Goal point is inside an obstacle")

    # Возможные направления движения (8-связность)
    directions = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1),           (0, 1),
                  (1, -1),  (1, 0),  (1, 1)]

    # Инициализация
    open_set = []
    heapq.heappush(open_set, (0, Node(start_grid[0], start_grid[1], 0)))

    came_from = {}
    g_score = {start_grid: 0}
    f_score = {start_grid: heuristic(start_grid, goal_grid)}

    closed_set = set()

    while open_set:
        _, current = heapq.heappop(open_set)
        current_pos = (current.x, current.y)

        if current_pos == goal_grid:
            # Путь найден, восстанавливаем его
            path = []
            while current:
                # Преобразуем обратно в мировые координаты
                world_x = x0 + current.x * grid_resolution
                world_y = y0 + current.y * grid_resolution
                path.append((world_x, world_y))
                current = current.parent
            return path[::-1]  # Разворачиваем путь (от старта к цели)

        closed_set.add(current_pos)

        for dx, dy in directions:
            neighbor_pos = (current.x + dx, current.y + dy)

            # Проверяем границы
            if not (0 <= neighbor_pos[0] < width and 0 <= neighbor_pos[1] < height):
                continue

            # Проверяем, что ячейка свободна
            if occupancy_grid[neighbor_pos[1], neighbor_pos[0]] == 1:
                continue

            # Стоимость перехода (диагональные движения дороже)
            move_cost = np.sqrt(dx**2 + dy**2) * grid_resolution
            tentative_g_score = g_score[current_pos] + move_cost

            if neighbor_pos in closed_set and tentative_g_score >= g_score.get(neighbor_pos, float('inf')):
                continue

            if tentative_g_score < g_score.get(neighbor_pos, float('inf')):
                # Этот путь лучше предыдущего
                came_from[neighbor_pos] = current_pos
                g_score[neighbor_pos] = tentative_g_score
                f_score[neighbor_pos] = tentative_g_score + \
                    heuristic(neighbor_pos, goal_grid)

                # Создаем новый узел
                neighbor_node = Node(neighbor_pos[0], neighbor_pos[1],
                                     f_score[neighbor_pos],
                                     Node(current.x, current.y, g_score[current_pos], current.parent))

                heapq.heappush(
                    open_set, (f_score[neighbor_pos], neighbor_node))

    # Путь не найден
    raise RuntimeError("No path found from start to goal")


def smooth_path(path: List[Tuple[float, float]],
                alpha: float = 0.5,
                beta: float = 0.3,
                iterations: int = 50) -> List[Tuple[float, float]]:
    """
    Сглаживание пути с помощью алгоритма градиентного спуска.

    :param path: исходный путь
    :param alpha: коэффициент сглаживания
    :param beta: коэффициент притяжения к исходным точкам
    :param iterations: количество итераций

    :return: сглаженный путь
    """
    if len(path) <= 2:
        return path

    smoothed_path = path.copy()

    for _ in range(iterations):
        for i in range(1, len(path) - 1):
            # Притяжение к соседним точкам
            smoothed_path[i] = (
                smoothed_path[i][0] + alpha * (smoothed_path[i-1][0] +
                                               smoothed_path[i+1][0] - 2 * smoothed_path[i][0]),
                smoothed_path[i][1] + alpha * (smoothed_path[i-1][1] +
                                               smoothed_path[i+1][1] - 2 * smoothed_path[i][1])
            )

            # Притяжение к исходным точкам
            smoothed_path[i] = (
                smoothed_path[i][0] + beta *
                (path[i][0] - smoothed_path[i][0]),
                smoothed_path[i][1] + beta * (path[i][1] - smoothed_path[i][1])
            )

    return smoothed_path


def generate_reference_trajectory(start: Tuple[float, float],
                                  goal: Tuple[float, float],
                                  occupancy_grid: np.ndarray,
                                  grid_resolution: float,
                                  grid_origin: Tuple[float,
                                                     float] = (0.0, 0.0),
                                  num_points: int = 50,
                                  smooth: bool = True) -> np.ndarray:
    """
    Генерирует опорную траекторию с использованием алгоритма A*.

    :param start: начальная точка (x, y)
    :param goal: целевая точка (x, y)
    :param occupancy_grid: матрица Occupancy Grid
    :param grid_resolution: разрешение сетки
    :param grid_origin: начало координат сетки
    :param num_points: количество точек в траектории
    :param smooth: применять ли сглаживание

    :return: опорная траектория в формате (N, 5) [x, y, v, theta, delta]
    """
    # Находим путь с помощью A*
    path_points = a_star_search(occupancy_grid, start, goal,
                                grid_resolution, grid_origin)

    # Сглаживаем путь
    if smooth and len(path_points) > 2:
        path_points = smooth_path(path_points)

    # Если нужно меньше точек, производим ресемплирование
    if len(path_points) > num_points:
        indices = np.linspace(0, len(path_points) - 1, num_points, dtype=int)
        path_points = [path_points[i] for i in indices]
    elif len(path_points) < num_points:
        # Интерполируем для получения большего количества точек
        from scipy.interpolate import interp1d
        x_vals = [p[0] for p in path_points]
        y_vals = [p[1] for p in path_points]

        # Параметрическая интерполяция
        t = np.linspace(0, 1, len(path_points))
        t_new = np.linspace(0, 1, num_points)

        fx = interp1d(t, x_vals, kind='cubic')
        fy = interp1d(t, y_vals, kind='cubic')

        x_new = fx(t_new)
        y_new = fy(t_new)
        path_points = list(zip(x_new, y_new))

    # Вычисляем ориентацию для каждой точки
    theta_points = np.zeros(len(path_points))
    for i in range(1, len(path_points) - 1):
        dx = path_points[i + 1][0] - path_points[i - 1][0]
        dy = path_points[i + 1][1] - path_points[i - 1][1]
        theta_points[i] = np.arctan2(dy, dx)

    # Первая и последняя точки
    if len(path_points) > 1:
        dx = path_points[1][0] - path_points[0][0]
        dy = path_points[1][1] - path_points[0][1]
        theta_points[0] = np.arctan2(dy, dx)

        dx = path_points[-1][0] - path_points[-2][0]
        dy = path_points[-1][1] - path_points[-2][1]
        theta_points[-1] = np.arctan2(dy, dx)

    # Создаем опорную траекторию
    ref_traj = np.zeros((len(path_points), 5))
    ref_traj[:, 0] = [p[0] for p in path_points]  # x
    ref_traj[:, 1] = [p[1] for p in path_points]  # y
    ref_traj[:, 2] = 1.0  # постоянная скорость
    ref_traj[:, 3] = theta_points  # направление движения
    ref_traj[:, 4] = 0.0  # нулевой угол поворота

    return ref_traj


def generate_avoidance_trajectory(start: Tuple[float, float],
                                  goal: Tuple[float, float],
                                  occupancy_grid: np.ndarray,
                                  grid_resolution: float,
                                  grid_origin: Tuple[float,
                                                     float] = (0.0, 0.0),
                                  num_points: int = 50) -> np.ndarray:
    """
    Алиас для generate_reference_trajectory для обратной совместимости.
    """
    return generate_reference_trajectory(start, goal, occupancy_grid,
                                         grid_resolution, grid_origin, num_points)


def plot_trajectory(trajectory: np.ndarray,
                    occupancy_grid: Optional[np.ndarray] = None,
                    grid_resolution: float = 0.1,
                    grid_origin: Tuple[float, float] = (0.0, 0.0),
                    title: str = "Trajectory"):
    """
    Визуализирует траекторию с возможностью отображения Occupancy Grid.

    :param trajectory: траектория в формате (N, 5) [x, y, v, theta, delta]
    :param occupancy_grid: матрица Occupancy Grid (опционально)
    :param grid_resolution: разрешение сетки
    :param grid_origin: начало координат сетки
    :param title: заголовок графика
    """
    plt.figure(figsize=(12, 10))

    if occupancy_grid is not None:
        # Отображаем Occupancy Grid
        x0, y0 = grid_origin
        extent = [x0, x0 + occupancy_grid.shape[1] * grid_resolution,
                  y0, y0 + occupancy_grid.shape[0] * grid_resolution]

        plt.imshow(occupancy_grid, origin='lower', extent=extent,
                   cmap='binary', alpha=0.7, vmin=0, vmax=1)

    # Рисуем траекторию
    plt.plot(trajectory[:, 0], trajectory[:, 1],
             'b-', label='Trajectory', linewidth=2)
    plt.plot(trajectory[:, 0], trajectory[:, 1],
             'ro', markersize=3, label='Points')

    # Рисуем стрелки направления
    for i in range(0, len(trajectory), max(1, len(trajectory)//10)):
        x, y, v, theta, delta = trajectory[i]
        dx = 0.5 * np.cos(theta)
        dy = 0.5 * np.sin(theta)
        plt.arrow(x, y, dx, dy, head_width=0.2, head_length=0.3,
                  fc='green', ec='green', alpha=0.7)

    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()
