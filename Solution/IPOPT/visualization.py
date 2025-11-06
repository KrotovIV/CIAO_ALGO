import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict
from matplotlib.patches import Circle, Rectangle
from occupancy_grid_utils import visualize_occupancy_grid_with_circles
from obstacle import CircularObstacle


def plot_optimization_results(boundary_square: Tuple,
                              states_opt: np.ndarray,
                              controls_opt: np.ndarray,
                              ref_traj: np.ndarray,
                              circles: List[Tuple[float, float, float]],
                              occupancy_grid: Optional[np.ndarray] = None,
                              grid_resolution: float = 0.1,
                              grid_origin: Tuple[float, float] = (0.0, 0.0),
                              obstacles: Optional[List[Tuple[float,
                                                             float, float, float]]] = None,
                              title: str = "Trajectory Optimization Results",
                              L: float = 2.5,
                              ):
    """
    Визуализирует результаты оптимизации траектории.

    :param states_opt: оптимизированные состояния (N+1, 5)
    :param controls_opt: оптимизированные управления (N, 2)
    :param ref_traj: опорная траектория (N+1, 5)
    :param circles: список ограничивающих кругов [(xo, yo, r), ...]
    :param occupancy_grid: матрица Occupancy Grid (опционально)
    :param grid_resolution: разрешение сетки (если используется occupancy_grid)
    :param grid_origin: начало координат сетки (если используется occupancy_grid)
    :param obstacles: список препятствий в формате (x, y, width, height) (опционально)
    :param title: заголовок графика
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(title, fontsize=16)

    # 1. Основной график: траектории и круги
    ax1 = axes[0, 0]
    plot_trajectories(ax1, states_opt, ref_traj, circles,
                      obstacles, boundary_square)
    ax1.set_title("Trajectories and Collision Avoidance Circles")
    ax1.legend()

    # 2. График состояний
    ax2 = axes[0, 1]
    plot_states(ax2, states_opt, ref_traj)
    ax2.set_title("State Variables")
    ax2.legend()

    # 3. График управлений
    ax3 = axes[1, 0]
    plot_controls(ax3, controls_opt)
    ax3.set_title("Control Variables")
    ax3.legend()

    # 4. Occupancy Grid (если предоставлен)
    ax4 = axes[1, 1]
    if occupancy_grid is not None:
        plot_occupancy_grid(ax4, occupancy_grid, grid_resolution, grid_origin,
                            states_opt, ref_traj, circles)
        ax4.set_title("Occupancy Grid with Trajectories")
    else:
        ax4.axis('off')
        ax4.text(0.5, 0.5, 'No Occupancy Grid provided',
                 horizontalalignment='center', verticalalignment='center',
                 transform=ax4.transAxes, fontsize=12)

    plt.tight_layout()
    plt.show()


def plot_trajectories(ax: plt.Axes,
                      states_opt: np.ndarray,
                      ref_traj: np.ndarray,
                      circles: List[Tuple[float, float, float]],
                      obstacles: Optional[List[CircularObstacle]] = None,
                      boundary_square: Optional[Tuple[float, float, float, float]] = None):
    """
    Рисует траектории и ограничивающие круги.

    :param ax: объект axes для рисования
    :param states_opt: оптимизированные состояния
    :param ref_traj: опорная траектория
    :param circles: список ограничивающих кругов
    :param obstacles: список препятствий (круглых или прямоугольных)
    :param boundary_square: границы квадрата
    """
    # Опорная траектория
    ax.plot(ref_traj[:, 0], ref_traj[:, 1], 'r--', linewidth=2,
            label='Reference Trajectory')
    ax.plot(ref_traj[:, 0], ref_traj[:, 1], 'ro', markersize=4, alpha=0.7)

    # Оптимизированная траектория
    ax.plot(states_opt[:, 0], states_opt[:, 1], 'b-', linewidth=3,
            label='Optimized Trajectory')
    ax.plot(states_opt[:, 0], states_opt[:, 1], 'bo', markersize=4, alpha=0.7)

    # Ограничивающие круги
    for i, (xo, yo, r) in enumerate(circles):
        circle = Circle((xo, yo), r, color='green', alpha=0.3,
                        fill=True, linestyle='-', linewidth=1)
        ax.add_patch(circle)
        if i % 5 == 0:  # Показываем только каждый 5-й круг для читаемости
            ax.plot(xo, yo, 'g+', markersize=8, markeredgewidth=2)

    # Препятствия (если предоставлены)
    if obstacles is not None:
        for i, obs in enumerate(obstacles):
            if isinstance(obs, CircularObstacle):
                circle = obs.to_circle_patch(
                    color='red', alpha=0.5,
                    label='Circular Obstacle' if i == 0 else ""
                )
                ax.add_patch(circle)

    # Направления движения для оптимизированной траектории
    for i in range(0, len(states_opt), max(1, len(states_opt)//10)):
        x, y, v, theta, delta = states_opt[i]
        dx = 0.5 * np.cos(theta)
        dy = 0.5 * np.sin(theta)
        ax.arrow(x, y, dx, dy, head_width=0.2, head_length=0.3,
                 fc='blue', ec='blue', alpha=0.7)

    # Квадрат с границами (если задан)
    if boundary_square is not None:
        min_x, max_x, min_y, max_y = boundary_square
        width = max_x - min_x
        height = max_y - min_y

        # Рисуем квадрат
        square = Rectangle((min_x, min_y), width, height,
                           fill=False, color='purple', linestyle='--', linewidth=2,
                           label='Boundary Square')
        ax.add_patch(square)

        # Проверяем, выходят ли круги за пределы квадрата
        for i, (xo, yo, r) in enumerate(circles):
            if (xo - r < min_x or xo + r > max_x or
                    yo - r < min_y or yo + r > max_y):
                print(f"Warning: Circle {i} at ({xo:.2f}, {yo:.2f}) with radius {r:.2f} "
                      f"extends beyond the boundary square")

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.grid(True)
    ax.axis('equal')


def plot_states(ax: plt.Axes, states_opt: np.ndarray, ref_traj: np.ndarray):
    """
    Рисует графики состояний.

    :param ax: объект axes для рисования
    :param states_opt: оптимизированные состояния
    :param ref_traj: опорная траектория
    """
    time = np.arange(len(states_opt))

    # Координаты X
    ax.plot(time, states_opt[:, 0], 'b-', label='X optimized', linewidth=2)
    ax.plot(time, ref_traj[:, 0], 'r--', label='X reference', alpha=0.7)

    # Координаты Y
    ax.plot(time, states_opt[:, 1], 'g-', label='Y optimized', linewidth=2)
    ax.plot(time, ref_traj[:, 1], 'm--', label='Y reference', alpha=0.7)

    # Скорость
    ax.plot(time, states_opt[:, 2], 'c-',
            label='Velocity optimized', linewidth=2)
    ax.plot(time, ref_traj[:, 2], 'y--', label='Velocity reference', alpha=0.7)

    # Угол направления
    ax.plot(time, states_opt[:, 3], 'k-', label='Theta optimized', linewidth=2)
    ax.plot(time, ref_traj[:, 3], 'gray', linestyle='--',
            label='Theta reference', alpha=0.7)

    ax.set_xlabel('Time step')
    ax.set_ylabel('State values')
    ax.grid(True)


def plot_controls(ax: plt.Axes, controls_opt: np.ndarray):
    """
    Рисует графики управлений.

    :param ax: объект axes для рисования
    :param controls_opt: оптимизированные управления
    """
    time = np.arange(len(controls_opt))

    # Ускорение
    ax.plot(time, controls_opt[:, 0], 'r-', label='Acceleration', linewidth=2)

    # Скорость поворота руля
    ax.plot(time, controls_opt[:, 1], 'b-', label='Steering rate', linewidth=2)

    ax.set_xlabel('Time step')
    ax.set_ylabel('Control values')
    ax.grid(True)
    ax.axhline(0, color='black', linestyle='-', alpha=0.3)


def plot_occupancy_grid(ax: plt.Axes,
                        occupancy_grid: np.ndarray,
                        grid_resolution: float,
                        grid_origin: Tuple[float, float],
                        states_opt: np.ndarray,
                        ref_traj: np.ndarray,
                        circles: List[Tuple[float, float, float]]):
    """
    Рисует Occupancy Grid с траекториями.

    :param ax: объект axes для рисования
    :param occupancy_grid: матрица Occupancy Grid
    :param grid_resolution: разрешение сетки
    :param grid_origin: начало координат сетки
    :param states_opt: оптимизированные состояния
    :param ref_traj: опорная траектория
    :param circles: список ограничивающих кругов
    """
    x0, y0 = grid_origin
    extent = [x0, x0 + occupancy_grid.shape[1] * grid_resolution,
              y0, y0 + occupancy_grid.shape[0] * grid_resolution]

    # Отображаем Occupancy Grid
    ax.imshow(occupancy_grid, origin='lower', extent=extent,
              cmap='binary', alpha=0.7, vmin=0, vmax=1)

    # Опорная траектория
    ax.plot(ref_traj[:, 0], ref_traj[:, 1], 'r--', linewidth=2,
            label='Reference Trajectory')

    # Оптимизированная траектория
    ax.plot(states_opt[:, 0], states_opt[:, 1], 'b-', linewidth=3,
            label='Optimized Trajectory')

    # Ограничивающие круги
    for xo, yo, r in circles:
        circle = Circle((xo, yo), r, color='green', alpha=0.3,
                        fill=True, linestyle='-', linewidth=1)
        ax.add_patch(circle)
        ax.plot(xo, yo, 'g+', markersize=6, markeredgewidth=1.5)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.legend()
    ax.grid(True)
    ax.axis('equal')


def plot_collision_constraints_violation(states_opt: np.ndarray,
                                         circles: List[Tuple[float, float, float]]):
    """
    Визуализирует нарушение ограничений столкновений.

    :param states_opt: оптимизированные состояния
    :param circles: список ограничивающих кругов
    """
    violations = []
    for i, state in enumerate(states_opt):
        x, y = state[0], state[1]
        xo, yo, r = circles[i]
        distance = np.sqrt((x - xo)**2 + (y - yo)**2)
        violation = max(0, distance - r)  # Положительное значение = нарушение
        violations.append(violation)

    plt.figure(figsize=(10, 6))
    plt.plot(violations, 'r-', linewidth=2)
    plt.axhline(0, color='green', linestyle='--', alpha=0.7)
    plt.xlabel('Time step')
    plt.ylabel('Constraint violation (m)')
    plt.title('Collision Constraints Violation')
    plt.grid(True)
    plt.show()


def create_animation(states_opt: np.ndarray,
                     ref_traj: np.ndarray,
                     circles: List[Tuple[float, float, float]],
                     obstacles: Optional[List[Tuple[float,
                                                    float, float, float]]] = None,
                     filename: str = 'trajectory_animation.gif'):
    """
    Создает анимацию движения по траектории.

    :param states_opt: оптимизированные состояния
    :param ref_traj: опорная траектория
    :param circles: список ограничивающих кругов
    :param obstacles: список препятствий (опционально)
    :param filename: имя файла для сохранения анимации
    """
    try:
        from matplotlib.animation import FuncAnimation
        import matplotlib.animation as animation

        fig, ax = plt.subplots(figsize=(10, 8))

        # Настройка графика
        ax.set_xlim(min(states_opt[:, 0].min(), ref_traj[:, 0].min()) - 2,
                    max(states_opt[:, 0].max(), ref_traj[:, 0].max()) + 2)
        ax.set_ylim(min(states_opt[:, 1].min(), ref_traj[:, 1].min()) - 2,
                    max(states_opt[:, 1].max(), ref_traj[:, 1].max()) + 2)

        # Рисуем статические элементы
        ax.plot(ref_traj[:, 0], ref_traj[:, 1], 'r--', label='Reference')
        ax.plot(states_opt[:, 0], states_opt[:, 1], 'b-', label='Optimized')

        # Препятствия
        if obstacles:
            for obs in obstacles:
                x, y, width, height = obs
                rect = Rectangle((x, y), width, height, color='red', alpha=0.5)
                ax.add_patch(rect)

        # Круги (только для визуального контекста)
        for xo, yo, r in circles[::5]:  # Каждый 5-й круг
            circle = Circle((xo, yo), r, color='green', alpha=0.2, fill=True)
            ax.add_patch(circle)

        # Динамические элементы
        robot_point, = ax.plot([], [], 'bo', markersize=10, label='Robot')
        robot_arrow = ax.arrow(0, 0, 0, 0, head_width=0.3, head_length=0.5,
                               fc='blue', ec='blue')

        ax.legend()
        ax.grid(True)
        ax.set_title('Trajectory Animation')
        ax.axis('equal')

        def update(frame):
            x, y, v, theta, delta = states_opt[frame]

            # Обновляем позицию робота
            robot_point.set_data([x], [y])

            # Обновляем стрелку направления
            ax.patches.remove(robot_arrow)
            robot_arrow = ax.arrow(x, y, 0.5*np.cos(theta), 0.5*np.sin(theta),
                                   head_width=0.3, head_length=0.5,
                                   fc='blue', ec='blue')
            ax.add_patch(robot_arrow)

            return robot_point, robot_arrow

        ani = FuncAnimation(fig, update, frames=len(states_opt),
                            interval=100, blit=True)

        # Сохраняем анимацию
        ani.save(filename, writer='pillow', fps=10)
        plt.close()
        print(f"Animation saved as {filename}")

    except ImportError:
        print("Animation requires matplotlib.animation module")
    except Exception as e:
        print(f"Error creating animation: {e}")
