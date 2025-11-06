import numpy as np
from typing import Tuple, List
from matplotlib.patches import Rectangle, Circle as MplCircle


class Obstacle:
    """
    Класс для представления прямоугольного препятствия.
    """

    def __init__(self, x: float, y: float, width: float, height: float):
        """
        Инициализирует препятствие.

        :param x: x-координата левого нижнего угла
        :param y: y-координата левого нижнего угла
        :param width: ширина препятствия
        :param height: высота препятствия
        """
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def contains_point(self, point_x: float, point_y: float) -> bool:
        """
        Проверяет, содержит ли препятствие точку.

        :param point_x: x-координата точки
        :param point_y: y-координата точки
        :return: True если точка внутри препятствия
        """
        return (self.x <= point_x <= self.x + self.width and
                self.y <= point_y <= self.y + self.height)

    def to_rectangle_patch(self, **kwargs) -> Rectangle:
        """
        Создает объект Rectangle для визуализации.

        :param kwargs: дополнительные параметры для Rectangle
        :return: объект Rectangle
        """
        return Rectangle((self.x, self.y), self.width, self.height, **kwargs)

    def to_tuple(self) -> Tuple[float, float, float, float]:
        """
        Возвращает представление препятствия в виде кортежа.

        :return: (x, y, width, height)
        """
        return (self.x, self.y, self.width, self.height)

    @staticmethod
    def from_tuple(obstacle_tuple: Tuple[float, float, float, float]) -> 'Obstacle':
        """
        Создает Obstacle из кортежа.

        :param obstacle_tuple: (x, y, width, height)
        :return: объект Obstacle
        """
        return Obstacle(*obstacle_tuple)

    @staticmethod
    def list_from_tuples(obstacle_tuples: List[Tuple[float, float, float, float]]) -> List['Obstacle']:
        """
        Создает список Obstacle из списка кортежей.

        :param obstacle_tuples: список кортежей (x, y, width, height)
        :return: список объектов Obstacle
        """
        return [Obstacle.from_tuple(t) for t in obstacle_tuples]

    @staticmethod
    def list_to_tuples(obstacles: List['Obstacle']) -> List[Tuple[float, float, float, float]]:
        """
        Преобразует список Obstacle в список кортежей.

        :param obstacles: список объектов Obstacle
        :return: список кортежей (x, y, width, height)
        """
        return [obs.to_tuple() for obs in obstacles]

    def __repr__(self) -> str:
        return f"Obstacle(x={self.x}, y={self.y}, width={self.width}, height={self.height})"


class CircularObstacle:
    """
    Класс для представления круглого препятствия.
    """

    def __init__(self, x: float, y: float, radius: float):
        """
        Инициализирует круглое препятствие.

        :param x: x-координата центра
        :param y: y-координата центра
        :param radius: радиус препятствия
        """
        self.x = x
        self.y = y
        self.radius = radius

    def contains_point(self, point_x: float, point_y: float) -> bool:
        """
        Проверяет, содержит ли препятствие точку.

        :param point_x: x-координата точки
        :param point_y: y-координата точки
        :return: True если точка внутри препятствия
        """
        distance = np.sqrt((point_x - self.x)**2 + (point_y - self.y)**2)
        return distance <= self.radius

    def to_circle_patch(self, **kwargs) -> MplCircle:
        """
        Создает объект Circle для визуализации.

        :param kwargs: дополнительные параметры для Circle
        :return: объект Circle
        """
        return MplCircle((self.x, self.y), self.radius, **kwargs)

    def to_tuple(self) -> Tuple[float, float, float]:
        """
        Возвращает представление препятствия в виде кортежа.

        :return: (x, y, radius)
        """
        return (self.x, self.y, self.radius)

    @staticmethod
    def from_tuple(obstacle_tuple: Tuple[float, float, float]) -> 'CircularObstacle':
        """
        Создает CircularObstacle из кортежа.

        :param obstacle_tuple: (x, y, radius)
        :return: объект CircularObstacle
        """
        return CircularObstacle(*obstacle_tuple)

    @staticmethod
    def list_from_tuples(obstacle_tuples: List[Tuple[float, float, float]]) -> List['CircularObstacle']:
        """
        Создает список CircularObstacle из списка кортежей.

        :param obstacle_tuples: список кортежей (x, y, radius)
        :return: список объектов CircularObstacle
        """
        return [CircularObstacle.from_tuple(t) for t in obstacle_tuples]

    @staticmethod
    def list_to_tuples(obstacles: List['CircularObstacle']) -> List[Tuple[float, float, float]]:
        """
        Преобразует список CircularObstacle в список кортежей.

        :param obstacles: список объектов CircularObstacle
        :return: список кортежей (x, y, radius)
        """
        return [obs.to_tuple() for obs in obstacles]

    def __repr__(self) -> str:
        return f"CircularObstacle(x={self.x}, y={self.y}, radius={self.radius})"

# Для обратной совместимости оставляем старый класс


class RectangularObstacle(Obstacle):
    """
    Класс для представления прямоугольного препятствия.
    """

    def __init__(self, x: float, y: float, width: float, height: float):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def contains_point(self, point_x: float, point_y: float) -> bool:
        return (self.x <= point_x <= self.x + self.width and
                self.y <= point_y <= self.y + self.height)

    def to_rectangle_patch(self, **kwargs) -> Rectangle:
        return Rectangle((self.x, self.y), self.width, self.height, **kwargs)

    def to_tuple(self) -> Tuple[float, float, float, float]:
        return (self.x, self.y, self.width, self.height)

    def __repr__(self) -> str:
        return f"RectangularObstacle(x={self.x}, y={self.y}, width={self.width}, height={self.height})"
