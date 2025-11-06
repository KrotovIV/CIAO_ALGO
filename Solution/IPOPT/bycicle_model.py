import numpy as np


def bicycle_model_rhs(state: np.ndarray, control: np.ndarray, L: float) -> np.ndarray:
    """
    Вычисляет правые части дифференциальных уравнений для велосипедной модели.

    :param state: вектор состояния [x, y, v, theta, delta]
    :param control: вектор управления [a, omega]
    :param L: расстояние между осями (колесная база)

    :return: производная состояния [dx_dt, dy_dt, dv_dt, dtheta_dt, ddelta_dt]
    """
    x, y, v, theta, delta = state
    a, omega = control

    dx_dt = v * np.cos(theta)
    dy_dt = v * np.sin(theta)
    dv_dt = a
    dtheta_dt = (v / L) * np.tan(delta)
    ddelta_dt = omega

    return np.array([dx_dt, dy_dt, dv_dt, dtheta_dt, ddelta_dt])


def integrate_bicycle_model(state: np.ndarray, control: np.ndarray, dt: float, L: float) -> np.ndarray:
    """
    Интегрирует велосипедную модель на одном шаге длиной dt методом Эйлера.

    :param state: текущее состояние [x, y, v, theta, delta]
    :param control: управление на шаге [a, omega]
    :param dt: длительность шага
    :param L: колесная база

    :return: состояние на следующем шаге
    """
    rhs = bicycle_model_rhs(state, control, L)
    next_state = state + dt * rhs
    return next_state


def simulate_trajectory(x0: np.ndarray, controls: list, dt: float, L: float) -> np.ndarray:
    """
    Симулирует траекторию для заданной последовательности управлений.

    :param x0: начальное состояние (вектор длины 5)
    :param controls: список управлений (каждое — вектор [a, omega])
    :param dt: длительность одного шага
    :param L: колесная база

    :return: массив состояний на каждом шаге (размер: (len(controls)+1) x 5)
    """
    n_steps = len(controls)
    states = np.zeros((n_steps + 1, len(x0)))
    states[0] = x0

    for i in range(n_steps):
        states[i+1] = integrate_bicycle_model(states[i], controls[i], dt, L)

    return states
