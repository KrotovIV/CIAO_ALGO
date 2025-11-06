import numpy as np
from typing import List, Tuple, Optional, Union
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.morphology import disk, dilation
from obstacle import CircularObstacle


def occupancy_grid_to_circles(occupancy_grid: np.ndarray,
                              ref_traj: np.ndarray,
                              grid_resolution: float,
                              grid_origin: Tuple[float, float] = (0.0, 0.0),
                              min_radius: float = 0.3,
                              max_radius: float = float('inf'),
                              dilation_radius: int = 2,
                              boundary_square: Optional[Tuple[float,
                                                              float, float, float]] = None,
                              search_radius_pixels: int = 10,
                              proximity_weight: float = 0.3) -> List[Tuple[float, float, float]]:
    """
    Преобразует Occupancy Grid в список ограничивающих кругов для траектории.
    Находит оптимальный центр, балансируя между большим радиусом и близостью к опорной траектории.
    """
    # Расширяем препятствия для создания буфера безопасности
    if dilation_radius > 0:
        dilated_grid = dilation(occupancy_grid, disk(dilation_radius))
    else:
        dilated_grid = occupancy_grid.copy()

    circles = []
    x0, y0 = grid_origin
    height, width = dilated_grid.shape

    # Если boundary_square не задан, вычисляем его из траектории
    if boundary_square is None:
        min_x = np.min(ref_traj[:, 0])
        max_x = np.max(ref_traj[:, 0])
        min_y = np.min(ref_traj[:, 1])
        max_y = np.max(ref_traj[:, 1])
        L = 2.5
        boundary_square = (min_x - L, max_x + L, min_y - L, max_y + L)

    min_x_bound, max_x_bound, min_y_bound, max_y_bound = boundary_square

    # Предварительно вычисляем преобразование расстояния для всей сетки
    from scipy.ndimage import distance_transform_edt
    dist_transform = distance_transform_edt(dilated_grid == 0)

    for i in range(len(ref_traj)):
        x_ref = ref_traj[i, 0]
        y_ref = ref_traj[i, 1]

        # Преобразуем мировые координаты в индексы сетки
        grid_x = int((x_ref - x0) / grid_resolution)
        grid_y = int((y_ref - y0) / grid_resolution)

        # Всегда создаем круг, даже если точка вне сетки
        if (0 <= grid_x < width and 0 <= grid_y < height):

            # Ищем оптимальный центр в окрестности с учетом близости к опорной траектории
            best_score = -float('inf')
            best_x_grid = grid_x
            best_y_grid = grid_y
            found_valid_center = False

            # Вычисляем максимальное возможное расстояние в окрестности для нормализации
            max_possible_dist = np.sqrt(
                2) * search_radius_pixels * grid_resolution

            # Ищем точку с лучшим компромиссом между радиусом и близостью к опорной линии
            for dy in range(-search_radius_pixels, search_radius_pixels + 1):
                for dx in range(-search_radius_pixels, search_radius_pixels + 1):
                    x_test = grid_x + dx
                    y_test = grid_y + dy

                    if (0 <= x_test < width and 0 <= y_test < height and
                            dilated_grid[y_test, x_test] == 0):  # Только свободные клетки

                        current_dist = dist_transform[y_test,
                                                      x_test] * grid_resolution

                        # Расстояние от тестовой точки до опорной точки
                        dist_to_ref = np.sqrt(dx**2 + dy**2) * grid_resolution

                        # Нормализованный радиус (0-1)
                        normalized_radius = min(
                            current_dist / (max_possible_dist * 0.5), 1.0)

                        # Нормализованное расстояние до опорной точки (0-1)
                        normalized_proximity = 1.0 - \
                            min(dist_to_ref / max_possible_dist, 1.0)

                        # Компромиссный score: радиус + близость к опорной линии
                        score = normalized_radius + proximity_weight * normalized_proximity

                        if score > best_score:
                            best_score = score
                            best_x_grid = x_test
                            best_y_grid = y_test
                            found_valid_center = True

            # Если не нашли валидный центр в окрестности, используем точку траектории
            if not found_valid_center:
                best_x_grid = grid_x
                best_y_grid = grid_y
                max_dist = dist_transform[best_y_grid, best_x_grid] * \
                    grid_resolution if dilated_grid[best_y_grid,
                                                    best_x_grid] == 0 else min_radius

            # Преобразуем обратно в мировые координаты
            x_center = x0 + best_x_grid * grid_resolution
            y_center = y0 + best_y_grid * grid_resolution

            # Вычисляем максимальный радиус от этого центра
            max_radius_obstacle = dist_transform[best_y_grid,
                                                 best_x_grid] * grid_resolution

            # Ограничиваем радиус границами квадрата
            dist_to_left = x_center - min_x_bound
            dist_to_right = max_x_bound - x_center
            dist_to_bottom = y_center - min_y_bound
            dist_to_top = max_y_bound - y_center

            max_radius_boundary = min(
                dist_to_left, dist_to_right, dist_to_bottom, dist_to_top)

            # Берем минимальный из двух ограничений
            radius = min(max_radius_obstacle, max_radius_boundary)

            # Гарантируем минимальный радиус безопасности
            radius = max(radius, min_radius)

            # Ограничиваем максимальным радиусом
            if max_radius != float('inf'):
                radius = min(radius, max_radius)

        else:
            # Если точка вне сетки, используем точку траектории с минимальным радиусом
            x_center = x_ref
            y_center = y_ref
            radius = min_radius

        circles.append((x_center, y_center, radius))

    return circles, boundary_square


def find_max_radius_at_point(occupancy_grid: np.ndarray,
                             point_x: int,
                             point_y: int,
                             grid_resolution: float) -> float:
    """
    Находит максимальный радиус круга без препятствий в заданной точке.
    Использует Евклидово расстояние преобразование.
    """
    height, width = occupancy_grid.shape

    # Проверяем, что точка внутри сетки
    if not (0 <= point_x < width and 0 <= point_y < height):
        return 0.0

    # Если точка занята, радиус = 0
    if occupancy_grid[point_y, point_x] == 1:
        return 0.0

    # Используем преобразование расстояния для нахождения максимального радиуса
    # Создаем бинарную карту: 0 - свободно, 1 - занято
    binary_map = occupancy_grid.copy()

    # Вычисляем преобразование расстояния
    from scipy.ndimage import distance_transform_edt
    dist_transform = distance_transform_edt(binary_map == 0)

    # Возвращаем расстояние в текущей точке (в метрах)
    return dist_transform[point_y, point_x] * grid_resolution


def visualize_occupancy_grid_with_circles(occupancy_grid: np.ndarray,
                                          circles: List[Tuple[float, float, float]],
                                          ref_traj: np.ndarray,
                                          grid_resolution: float,
                                          grid_origin: Tuple[float, float] = (
                                              0.0, 0.0),
                                          title: str = "Occupancy Grid with CIAO Circles"):
    """
    Визуализирует Occupancy Grid с ограничивающими кругами и траекторией.

    :param occupancy_grid: матрица Occupancy Grid
    :param circles: список ограничивающих кругов
    :param ref_traj: опорная траектория
    :param grid_resolution: размер ячейки сетки
    :param grid_origin: начало координат сетки
    :param title: заголовок графика
    """
    fig, ax = plt.subplots(figsize=(12, 10))

    # Отображаем Occupancy Grid
    x0, y0 = grid_origin
    extent = [x0, x0 + occupancy_grid.shape[1] * grid_resolution,
              y0, y0 + occupancy_grid.shape[0] * grid_resolution]

    ax.imshow(occupancy_grid, origin='lower', extent=extent,
              cmap='binary', alpha=0.7, vmin=0, vmax=1)

    # Отображаем опорную траекторию
    ax.plot(ref_traj[:, 0], ref_traj[:, 1], 'r-',
            linewidth=2, label='Reference Trajectory')
    ax.plot(ref_traj[:, 0], ref_traj[:, 1], 'ro', markersize=3)

    # Отображаем ограничивающие круги
    for i, (xo, yo, r) in enumerate(circles):
        circle = plt.Circle((xo, yo), r, color='blue', alpha=0.3,
                            fill=True, linestyle='-', linewidth=1)
        ax.add_patch(circle)
        ax.plot(xo, yo, 'bo', markersize=2)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    ax.axis('equal')
    plt.tight_layout()
    plt.show()


def create_simple_occupancy_grid(width: int = 50,
                                 height: int = 50,
                                 resolution: float = 0.5,
                                 obstacles: Optional[List[CircularObstacle]] = None) -> np.ndarray:
    """
    Создает простой Occupancy Grid для тестирования.

    :param width: ширина сетки в ячейках
    :param height: высота сетки в ячейках
    :param resolution: размер ячейки в метрах
    :param obstacles: список препятствий (круглых или прямоугольных)

    :return: матрица Occupancy Grid
    """
    grid = np.zeros((height, width), dtype=np.uint8)

    if obstacles is None:
        # Добавляем несколько тестовых круглых препятствий
        obstacles = [
            CircularObstacle(5.0, 5.0, 2.0),    # Круглое препятствие
            CircularObstacle(15.0, 10.0, 3.0),  # Еще одно круглое препятствие
            CircularObstacle(25.0, 20.0, 1.5)   # Круглое препятствие
        ]

    for obs in obstacles:
        if isinstance(obs, CircularObstacle):
            # Для круглых препятствий
            center_x = int(obs.x / resolution)
            center_y = int(obs.y / resolution)
            radius_pixels = int(obs.radius / resolution)

            # Заполняем круг в сетке
            y_indices, x_indices = np.ogrid[:height, :width]
            distance = np.sqrt((x_indices - center_x)**2 +
                               (y_indices - center_y)**2)
            grid[distance <= radius_pixels] = 1
    return grid


def verify_circles_avoid_obstacles(circles: List[Tuple[float, float, float]],
                                   occupancy_grid: np.ndarray,
                                   grid_resolution: float,
                                   grid_origin: Tuple[float, float] = (0.0, 0.0)) -> bool:
    """
    Проверяет, что все круги не пересекают препятствия.
    """
    x0, y0 = grid_origin
    height, width = occupancy_grid.shape

    for i, (x_center, y_center, radius) in enumerate(circles):
        # Преобразуем центр круга в координаты сетки
        center_x_grid = int((x_center - x0) / grid_resolution)
        center_y_grid = int((y_center - y0) / grid_resolution)

        # Преобразуем радиус в пиксели
        radius_pixels = int(radius / grid_resolution)

        # Проверяем все точки в круге
        for dx in range(-radius_pixels, radius_pixels + 1):
            for dy in range(-radius_pixels, radius_pixels + 1):
                if dx*dx + dy*dy <= radius_pixels*radius_pixels:  # Точка внутри круга
                    x_test = center_x_grid + dx
                    y_test = center_y_grid + dy

                    if (0 <= x_test < width and 0 <= y_test < height):
                        if occupancy_grid[y_test, x_test] == 1:
                            print(f"Warning: Circle {i} at ({x_center:.2f}, {y_center:.2f}) "
                                  f"with radius {radius:.2f} intersects obstacle at "
                                  f"grid cell ({x_test}, {y_test})")
                            return False
    return True
