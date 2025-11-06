# obstacle_editor.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from matplotlib.patches import Circle, Rectangle
from typing import List, Tuple, Optional, Callable, Dict
from obstacle import CircularObstacle


class ObstacleEditor:
    def __init__(self, boundary_square: Tuple[float, float, float, float],
                 grid_resolution: float = 0.1,
                 initial_obstacles: Optional[List[CircularObstacle]] = None,
                 start: Tuple = (None, None),
                 goal: Tuple = (None, None),
                 L: float = None):

        self.boundary_square = boundary_square
        self.grid_resolution = grid_resolution
        self.obstacles = initial_obstacles or []
        self.current_radius = 1.0
        self.selected_obstacle = None
        self.dragging = False
        self.should_optimize = False
        self.should_exit = False

        # Инициализируем instruction_text как None
        self.instruction_text = None

        # Создаем фигуру и оси
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        self.fig.subplots_adjust(bottom=0.2)

        # Устанавливаем заголовок окна
        self.fig.canvas.manager.set_window_title('Obstacle Editor')

        # Настройка области отображения
        min_x, max_x, min_y, max_y = boundary_square
        self.ax.set_xlim(min_x - 1, max_x + 1)
        self.ax.set_ylim(min_y - 1, max_y + 1)
        self.ax.set_aspect('equal')
        self.ax.grid(True)
        self.ax.set_title(
            'Obstacle Editor - Left:Add, Right:Delete, Drag:Move, +/-:Radius')

        # Рисуем boundary square
        self.boundary_patch = Rectangle(
            (min_x, min_y), max_x - min_x, max_y - min_y,
            fill=False, color='purple', linestyle='--', linewidth=2,
            label='Boundary Square'
        )
        self.ax.add_patch(self.boundary_patch)

        # Рисуем начальную и конечную точки
        if start[0] is not None and start[1] is not None:
            self.draw_circle(*start, L/4, color='green', label='Start')
        if goal[0] is not None and goal[1] is not None:
            self.draw_circle(*goal, L/4, color='blue', label='Goal')

        # Рисуем существующие препятствия
        self.obstacle_patches = []
        self.draw_obstacles()

        # Создаем элементы управления
        self.create_controls()

        # Подключаем обработчики событий
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.fig.canvas.mpl_connect('close_event', self.on_close)

    def create_controls(self):
        """Создает элементы управления интерфейсом."""
        # Слайдер для радиуса
        ax_radius = plt.axes([0.2, 0.1, 0.6, 0.03])
        self.radius_slider = Slider(
            ax_radius, 'Radius', 0.3, 3.0,
            valinit=self.current_radius, valstep=0.1
        )
        self.radius_slider.on_changed(self.update_radius)

        # Кнопки
        ax_clear = plt.axes([0.1, 0.05, 0.2, 0.04])
        ax_optimize = plt.axes([0.4, 0.05, 0.2, 0.04])
        ax_exit = plt.axes([0.7, 0.05, 0.2, 0.04])

        self.clear_button = Button(ax_clear, 'Clear All')
        self.optimize_button = Button(ax_optimize, 'Run Optimization')
        self.exit_button = Button(ax_exit, 'Exit Program')

        self.clear_button.on_clicked(self.clear_obstacles)
        self.optimize_button.on_clicked(self.run_optimization)
        self.exit_button.on_clicked(self.exit_program)

        # Текстовое поле с инструкциями (создаем только один раз)
        if self.instruction_text is None:
            self.instruction_text = self.ax.text(
                0.02, 0.98,
                'Instructions:\n'
                '- Left click: Add obstacle\n'
                '- Right click: Delete obstacle\n'
                '- Drag: Move obstacle\n'
                '- +/- keys: Adjust radius\n'
                f'- Current obstacles: {len(self.obstacles)}',
                transform=self.ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontsize=9
            )
        else:
            # Если уже существует, просто обновляем текст
            self.update_instructions()

    def draw_circle(self, x: float, y: float, radius: float, color: str = 'green', label: str = ''):
        """Рисует круг для старта/цели"""
        patch = Circle((x, y), radius, color=color,
                       alpha=0.7, fill=True, label=label)
        self.ax.add_patch(patch)
        self.fig.canvas.draw()

    def update_instructions(self):
        """Обновляет текст инструкций"""
        if self.instruction_text is not None:
            self.instruction_text.set_text(
                'Instructions:\n'
                '- Left click: Add obstacle\n'
                '- Right click: Delete obstacle\n'
                '- Drag: Move obstacle\n'
                '- +/- keys: Adjust radius\n'
                f'- Current obstacles: {len(self.obstacles)}'
            )
            self.fig.canvas.draw()

    def draw_obstacles(self):
        """Отрисовывает все препятствия."""
        # Удаляем старые патчи
        for patch in self.obstacle_patches:
            try:
                if patch in self.ax.patches:
                    patch.remove()
                elif hasattr(patch, 'remove') and patch in self.ax.texts:
                    patch.remove()
            except:
                pass  # Игнорируем ошибки при удалении

        self.obstacle_patches = []

        # Рисуем новые препятствия
        for i, obstacle in enumerate(self.obstacles):
            color = 'red' if obstacle == self.selected_obstacle else 'darkred'
            patch = Circle(
                (obstacle.x, obstacle.y), obstacle.radius,
                color=color, alpha=0.7, fill=True
            )
            self.obstacle_patches.append(patch)
            self.ax.add_patch(patch)

            # Добавляем текст с номером
            text = self.ax.text(obstacle.x, obstacle.y, str(i+1),
                                ha='center', va='center', color='white',
                                fontweight='bold', fontsize=8)
            self.obstacle_patches.append(text)

        self.update_instructions()
        self.fig.canvas.draw()

    def is_point_in_boundary(self, x: float, y: float) -> bool:
        """Проверяет, находится ли точка внутри boundary square."""
        min_x, max_x, min_y, max_y = self.boundary_square
        return min_x <= x <= max_x and min_y <= y <= max_y

    def get_obstacle_at_point(self, x: float, y: float) -> Optional[CircularObstacle]:
        """Находит препятствие в указанной точке."""
        for obstacle in reversed(self.obstacles):
            if obstacle.contains_point(x, y):
                return obstacle
        return None

    def on_click(self, event):
        """Обрабатывает клики мыши."""
        if event.inaxes != self.ax:
            return

        x, y = event.xdata, event.ydata

        if not self.is_point_in_boundary(x, y):
            return

        if event.button == 1:  # Левая кнопка
            obstacle = self.get_obstacle_at_point(x, y)
            if obstacle:
                # Выбираем существующее препятствие для перемещения
                self.selected_obstacle = obstacle
                self.dragging = True
                self.radius_slider.set_val(obstacle.radius)
            else:
                # Добавляем новое препятствие
                new_obstacle = CircularObstacle(x, y, self.current_radius)
                self.obstacles.append(new_obstacle)
                self.selected_obstacle = new_obstacle
                self.draw_obstacles()

        elif event.button == 3:  # Правая кнопка
            obstacle = self.get_obstacle_at_point(x, y)
            if obstacle:
                self.obstacles.remove(obstacle)
                if self.selected_obstacle == obstacle:
                    self.selected_obstacle = None
                self.draw_obstacles()

    def on_motion(self, event):
        """Обрабатывает движение мыши."""
        if event.inaxes != self.ax or not self.dragging or not self.selected_obstacle:
            return

        x, y = event.xdata, event.ydata

        if self.is_point_in_boundary(x, y):
            self.selected_obstacle.x = x
            self.selected_obstacle.y = y
            self.draw_obstacles()

    def on_release(self, event):
        """Обрабатывает отпускание кнопки мыши."""
        self.dragging = False

    def on_key_press(self, event):
        """Обрабатывает нажатия клавиш."""
        if event.key == '+' and self.selected_obstacle:
            self.selected_obstacle.radius = min(
                3.0, self.selected_obstacle.radius + 0.1)
            self.radius_slider.set_val(self.selected_obstacle.radius)
            self.draw_obstacles()
        elif event.key == '-' and self.selected_obstacle:
            self.selected_obstacle.radius = max(
                0.3, self.selected_obstacle.radius - 0.1)
            self.radius_slider.set_val(self.selected_obstacle.radius)
            self.draw_obstacles()

    def on_close(self, event):
        """Обрабатывает закрытие окна."""
        # При закрытии окна редактора завершаем программу
        self.should_exit = True

    def update_radius(self, val):
        """Обновляет текущий радиус."""
        self.current_radius = val
        if self.selected_obstacle:
            self.selected_obstacle.radius = val
            self.draw_obstacles()

    def clear_obstacles(self, event):
        """Очищает все препятствия."""
        self.obstacles = []
        self.selected_obstacle = None
        self.draw_obstacles()

    def run_optimization(self, event):
        """Запускает оптимизацию с текущими препятствиями."""
        self.should_optimize = True
        plt.close(self.fig)

    def exit_program(self, event):
        """Завершает программу."""
        self.should_exit = True
        plt.close(self.fig)

    def get_obstacles(self) -> List[CircularObstacle]:
        """Возвращает список препятствий."""
        return self.obstacles.copy()

    def show(self) -> Dict[str, any]:
        """Показывает редактор и возвращает результат."""
        plt.show()

        if self.should_optimize:
            return {'action': 'optimize', 'obstacles': self.get_obstacles()}
        elif self.should_exit:
            return {'action': 'exit', 'obstacles': self.get_obstacles()}
        else:
            return {'action': 'close', 'obstacles': self.get_obstacles()}


def create_obstacles_interactively(boundary_square: Tuple[float, float, float, float],
                                   grid_resolution: float = 0.1,
                                   initial_obstacles: Optional[List[CircularObstacle]] = None,
                                   start: Tuple = (None, None),
                                   goal: Tuple = (None, None),
                                   L: float = None) -> Dict[str, any]:
    """
    Создает препятствия в интерактивном режиме.
    """
    editor = ObstacleEditor(
        boundary_square, grid_resolution,
        initial_obstacles=initial_obstacles,
        start=start, goal=goal, L=L
    )
    result = editor.show()
    return result
