import numpy as np


def cost_function(X: np.ndarray, ref_traj: np.ndarray, weights: dict, N: int) -> float:
    """
    Вычисляет общую стоимость для траектории.

    :param X: вектор оптимизируемых переменных [x0, y0, v0, theta0, delta0, a0, omega0, x1, y1, ... , xN, yN, vN, thetaN, deltaN]
    :param ref_traj: Опорная траектория. Форма: (N+1, 5). Содержит [xref, yref, vref, thetaref, deltaref] для каждого состояния.
    :param weights: Словарь весовых коэффициентов. Ключи: 'wx', 'wy', 'wv', 'wtheta', 'wdelta', 'wa', 'womega'
    :param N: Количество шагов (сегментов) траектории. Количество состояний = N+1.

    :return: Общее значение функции стоимости.
    """
    # Извлечем веса для удобства
    wx = weights['wx']
    wy = weights['wy']
    wv = weights['wv']
    wtheta = weights['wtheta']
    wdelta = weights['wdelta']
    wa = weights['wa']
    womega = weights['womega']

    cost = 0.0

    # Размерность состояния и управления
    nx = 5
    nu = 2

    # Проходим по всем состояниям (от 0 до N)
    for i in range(N + 1):
        # Индексы текущего состояния в векторе X
        idx_x = i * (nx + nu)
        x_i = X[idx_x]     # x[i]
        y_i = X[idx_x + 1]  # y[i]
        v_i = X[idx_x + 2]  # v[i]
        theta_i = X[idx_x + 3]  # theta[i]
        delta_i = X[idx_x + 4]  # delta[i]

        # Опорные значения для i-го состояния
        xref_i = ref_traj[i, 0]
        yref_i = ref_traj[i, 1]
        vref_i = ref_traj[i, 2]
        thetaref_i = ref_traj[i, 3]
        deltaref_i = ref_traj[i, 4]

        # Добавляем к стоимости слагаемые, связанные с отклонением состояния
        cost += wx * (x_i - xref_i)**2
        cost += wy * (y_i - yref_i)**2
        cost += wv * (v_i - vref_i)**2
        cost += wtheta * (theta_i - thetaref_i)**2
        cost += wdelta * (delta_i - deltaref_i)**2

    # Проходим по всем управлениям (от 0 до N-1)
    for i in range(N):
        # Индексы текущего управления в векторе X
        idx_u = i * (nx + nu) + nx
        a_i = X[idx_u]     # a[i]
        omega_i = X[idx_u + 1]  # omega[i]

        # Добавляем к стоимости слагаемые, связанные с величиной управления
        cost += wa * (a_i)**2
        cost += womega * (omega_i)**2

    return cost


def cost_gradient(X: np.ndarray, ref_traj: np.ndarray, weights: dict, N: int) -> np.ndarray:
    """
    Вычисляет градиент функции стоимости.

    :param X: вектор оптимизируемых переменных.
    :param ref_traj: Опорная траектория. Форма: (N+1, 5).
    :param weights: Словарь весовых коэффициентов.
    :param N: Количество шагов (сегментов) траектории.

    :return: Градиент функции стоимости. Имеет ту же длину, что и X.
    """
    nx = 5
    nu = 2
    total_vars = N * (nx + nu) + nx
    grad = np.zeros(total_vars)

    wx = weights['wx']
    wy = weights['wy']
    wv = weights['wv']
    wtheta = weights['wtheta']
    wdelta = weights['wdelta']
    wa = weights['wa']
    womega = weights['womega']

    # Обрабатываем состояния
    for i in range(N + 1):
        idx = i * (nx + nu)
        x_i = X[idx]
        y_i = X[idx + 1]
        v_i = X[idx + 2]
        theta_i = X[idx + 3]
        delta_i = X[idx + 4]

        xref_i = ref_traj[i, 0]
        yref_i = ref_traj[i, 1]
        vref_i = ref_traj[i, 2]
        thetaref_i = ref_traj[i, 3]
        deltaref_i = ref_traj[i, 4]

        # Производные по состояниям
        grad[idx] += 2 * wx * (x_i - xref_i)        # dCost/dx_i
        grad[idx + 1] += 2 * wy * (y_i - yref_i)        # dCost/dy_i
        grad[idx + 2] += 2 * wv * (v_i - vref_i)        # dCost/dv_i
        grad[idx + 3] += 2 * wtheta * (theta_i - thetaref_i)  # dCost/dtheta_i
        grad[idx + 4] += 2 * wdelta * (delta_i - deltaref_i)  # dCost/ddelta_i

    # Обрабатываем управления
    for i in range(N):
        idx = i * (nx + nu) + nx  # Позиция управления в сегменте
        global_idx = i * (nx + nu) + nx  # Глобальная позиция в векторе X

        a_i = X[global_idx]
        omega_i = X[global_idx + 1]

        # Производные по управлениям
        grad[global_idx] += 2 * wa * a_i           # dCost/da_i
        grad[global_idx + 1] += 2 * womega * omega_i   # dCost/domega_i

    return grad
