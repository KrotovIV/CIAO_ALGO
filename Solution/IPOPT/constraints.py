import numpy as np
from typing import List, Tuple
from bycicle_model import integrate_bicycle_model


def collision_constraints(X: np.ndarray, circles: List[Tuple[float, float, float]],
                          N: int, nx: int = 5, nu: int = 2) -> np.ndarray:
    """
    Вычисляет значения ограничений-неравенств для избежания столкновений.

    :param X: вектор оптимизируемых переменных [x0, y0, v0, theta0, delta0, a0, omega0, x1, y1, ...]
    :param circles: список кортежей (xo, yo, r) для каждого шага, определяющих ограничивающие круги
    :param N: количество шагов (сегментов) траектории
    :param nx: размерность вектора состояния (по умолчанию 5)
    :param nu: размерность вектора управления (по умолчанию 2)

    :return: вектор значений ограничений (длина = N+1)
    """
    constraints = np.zeros(N + 1)

    for i in range(N + 1):
        # Индекс текущего состояния в векторе X
        idx = i * (nx + nu)
        x_i = X[idx]      # x[i]
        y_i = X[idx + 1]  # y[i]

        # Параметры ограничивающего круга для i-го шага
        xo_i, yo_i, r_i = circles[i]

        # Ограничение: (x_i - xo_i)^2 + (y_i - yo_i)^2 - r_i^2 <= 0
        distance_sq = (x_i - xo_i)**2 + (y_i - yo_i)**2
        constraints[i] = distance_sq - r_i**2

        # Отладочный вывод
        if distance_sq > r_i**2:
            print(f"Warning: Constraint violation at step {i}: "
                  f"distance={np.sqrt(distance_sq):.2f}, radius={r_i:.2f}")

    return constraints


def collision_constraints_jacobian(X: np.ndarray, circles: List[Tuple[float, float, float]],
                                   N: int, nx: int = 5, nu: int = 2) -> np.ndarray:
    """
    Вычисляет якобиан ограничений-неравенств для избежания столкновений.

    :param X: вектор оптимизируемых переменныхf
    :param circles: список кортежей (xo, yo, r) для каждого шага
    :param N: количество шагов (сегментов) траектории
    :param nx: размерность вектора состояния (по умолчанию 5)
    :param nu: размерность вектора управления (по умолчанию 2)

    :return: матрица Якоби (размер: (N+1) x (N*(nx+nu)+nx))
    """
    n_constraints = N + 1
    n_vars = N * (nx + nu) + nx
    jac = np.zeros((n_constraints, n_vars))

    for i in range(N + 1):
        # Индекс текущего состояния в векторе X
        idx = i * (nx + nu)
        x_i = X[idx]      # x[i]
        y_i = X[idx + 1]  # y[i]

        # Параметры ограничивающего круга для i-го шага
        xo_i, yo_i, r_i = circles[i]

        # Производные ограничения по координатам x и y
        jac[i, idx] = 2 * (x_i - xo_i)    # dConstraint/dx_i
        jac[i, idx + 1] = 2 * (y_i - yo_i)  # dConstraint/dy_i

    return jac


def dynamics_constraints(X: np.ndarray, dt: float, L: float,
                         N: int, nx: int = 5, nu: int = 2) -> np.ndarray:
    """
    Вычисляет ограничения динамики (равенства) для велосипедной модели.

    :param X: вектор оптимизируемых переменных
    :param dt: длительность шага
    :param L: колесная база
    :param N: количество шагов (сегментов) траектории
    :param nx: размерность вектора состояния (по умолчанию 5)
    :param nu: размерность вектора управления (по умолчанию 2)

    :return: вектор ограничений динамики (длина = N * nx)
    """
    constraints = np.zeros(N * nx)

    for i in range(N):
        # Текущее состояние
        idx_current = i * (nx + nu)
        state_current = X[idx_current:idx_current + nx]

        # Текущее управление
        control_current = X[idx_current + nx:idx_current + nx + nu]

        # Следующее состояние
        idx_next = (i + 1) * (nx + nu)
        state_next = X[idx_next:idx_next + nx]

        # Предсказанное следующее состояние по модели
        predicted_next = integrate_bicycle_model(
            state_current, control_current, dt, L)

        # Ограничение: predicted_next - state_next = 0
        constraints[i * nx:(i + 1) * nx] = predicted_next - state_next

    return constraints


def dynamics_constraints_jacobian(X: np.ndarray, dt: float, L: float,
                                  N: int, nx: int = 5, nu: int = 2) -> np.ndarray:
    """
    Вычисляет якобиан ограничений динамики для велосипедной модели.

    :param X: вектор оптимизируемых переменных
    :param dt: длительность шага
    :param L: колесная база
    :param N: количество шагов (сегментов) траектории
    :param nx: размерность вектора состояния (по умолчанию 5)
    :param nu: размерность вектора управления (по умолчанию 2)

    :return: матрица Якоби ограничений динамики
    """
    n_constraints = N * nx
    n_vars = N * (nx + nu) + nx
    jac = np.zeros((n_constraints, n_vars))

    for i in range(N):
        # Текущее состояние и управление
        idx_current = i * (nx + nu)
        state_current = X[idx_current:idx_current + nx]
        control_current = X[idx_current + nx:idx_current + nx + nu]

        # Вычисляем производные для модели
        x, y, v, theta, delta = state_current
        a, omega = control_current

        # Производные правой части модели
        dx_dx = 0
        dx_dy = 0
        dx_dv = np.cos(theta)
        dx_dtheta = -v * np.sin(theta)
        dx_ddelta = 0

        dy_dx = 0
        dy_dy = 0
        dy_dv = np.sin(theta)
        dy_dtheta = v * np.cos(theta)
        dy_ddelta = 0

        dv_dx = 0
        dv_dy = 0
        dv_dv = 0
        dv_dtheta = 0
        dv_ddelta = 0

        dtheta_dx = 0
        dtheta_dy = 0
        dtheta_dv = np.tan(delta) / L
        dtheta_dtheta = 0
        dtheta_ddelta = (v / L) * (1 / np.cos(delta)**2)

        ddelta_dx = 0
        ddelta_dy = 0
        ddelta_dv = 0
        ddelta_dtheta = 0
        ddelta_ddelta = 0

        # Матрица Якоби для правой части модели
        df_dstate = np.array([
            [dx_dx, dx_dy, dx_dv, dx_dtheta, dx_ddelta],
            [dy_dx, dy_dy, dy_dv, dy_dtheta, dy_ddelta],
            [dv_dx, dv_dy, dv_dv, dv_dtheta, dv_ddelta],
            [dtheta_dx, dtheta_dy, dtheta_dv, dtheta_dtheta, dtheta_ddelta],
            [ddelta_dx, ddelta_dy, ddelta_dv, ddelta_dtheta, ddelta_ddelta]
        ])

        df_dcontrol = np.array([
            [0, 0],      # dx/da, dx/domega
            [0, 0],      # dy/da, dy/domega
            [1, 0],      # dv/da, dv/domega
            [0, 0],      # dtheta/da, dtheta/domega
            [0, 1]       # ddelta/da, ddelta/domega
        ])

        # Матрица Якоби для метода Эйлера
        A = np.eye(nx) + dt * df_dstate
        B = dt * df_dcontrol

        # Заполняем якобиан для текущего шага
        constraint_start = i * nx
        constraint_end = (i + 1) * nx

        # Производные по текущему состоянию
        jac[constraint_start:constraint_end, idx_current:idx_current + nx] = A

        # Производные по текущему управлению
        jac[constraint_start:constraint_end,
            idx_current + nx:idx_current + nx + nu] = B

        # Производные по следующему состоянию
        idx_next = (i + 1) * (nx + nu)
        jac[constraint_start:constraint_end,
            idx_next:idx_next + nx] = -np.eye(nx)

    return jac
