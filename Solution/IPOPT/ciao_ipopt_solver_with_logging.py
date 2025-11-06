import numpy as np
from typing import List, Tuple, Dict, Optional, Callable
import cyipopt as ipopt
from constraints import collision_constraints, collision_constraints_jacobian, dynamics_constraints, dynamics_constraints_jacobian
from cost_functions import cost_function, cost_gradient
import json
import time
from datetime import datetime


class IterationLogger:
    """
    Класс для логирования данных итераций оптимизации.
    """

    def __init__(self, log_inner_newton: bool = True):
        self.log_inner_newton = log_inner_newton
        self.outer_iterations = []
        self.current_outer_iter = None
        self.start_time = time.time()

    def start_outer_iteration(self, iter_count: int, mu: float, obj_value: float,
                              inf_pr: float, inf_du: float, lg_norm: float,
                              alpha_du: float, alpha_pr: float, ls_trials: int):
        """
        Начинает логирование внешней итерации (μ-итерации).
        """
        self.current_outer_iter = {
            'iter': iter_count,
            'mu': mu,
            'obj_value': obj_value,
            'inf_pr': inf_pr,
            'inf_du': inf_du,
            'lg_norm': lg_norm,
            'alpha_du': alpha_du,
            'alpha_pr': alpha_pr,
            'ls_trials': ls_trials,
            'start_time': time.time() - self.start_time,
            'inner_iterations': [] if self.log_inner_newton else None
        }

    def add_inner_iteration(self, iter_count: int, obj_value: float, inf_pr: float,
                            inf_du: float, lg_norm: float, alpha_du: float,
                            alpha_pr: float, ls_trials: int):
        """
        Добавляет данные внутренней итерации (метод Ньютона).
        """
        if self.current_outer_iter is not None and self.log_inner_newton:
            inner_iter = {
                'inner_iter': iter_count,
                'obj_value': obj_value,
                'inf_pr': inf_pr,
                'inf_du': inf_du,
                'lg_norm': lg_norm,
                'alpha_du': alpha_du,
                'alpha_pr': alpha_pr,
                'ls_trials': ls_trials,
                'time': time.time() - self.start_time
            }
            self.current_outer_iter['inner_iterations'].append(inner_iter)

    def end_outer_iteration(self, success: bool = True):
        """
        Завершает логирование внешней итерации.
        """
        if self.current_outer_iter is not None:
            self.current_outer_iter['end_time'] = time.time() - self.start_time
            self.current_outer_iter['success'] = success
            self.outer_iterations.append(self.current_outer_iter)
            self.current_outer_iter = None

    def save_to_file(self, filename: str):
        """
        Сохраняет данные логирования в файл JSON.
        """
        data = {
            'log_timestamp': datetime.now().isoformat(),
            'log_inner_newton': self.log_inner_newton,
            'total_duration': time.time() - self.start_time,
            'outer_iterations': self.outer_iterations
        }

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

    def get_summary_stats(self) -> Dict:
        """
        Возвращает статистику по оптимизации.
        """
        if not self.outer_iterations:
            return {}

        return {
            'total_outer_iterations': len(self.outer_iterations),
            'total_inner_iterations': 0,
            'final_mu': self.outer_iterations[-1]['mu'],
            'final_objective': self.outer_iterations[-1]['obj_value'],
            'total_duration': time.time() - self.start_time
        }


class LoggingCIAOProblem:
    """
    Расширенная версия CIAOProblem с логированием итераций.
    """

    def __init__(self,
                 ref_traj: np.ndarray,
                 circles: List[Tuple[float, float, float]],
                 dt: float,
                 L: float,
                 weights: Dict[str, float],
                 x0: Optional[np.ndarray] = None,
                 log_inner_newton: bool = True):
        """
        Инициализирует задачу оптимизации с логированием.

        :param ref_traj: опорная траектория, форма (N+1, 5)
        :param circles: список ограничивающих кругов для каждого шага
        :param dt: длительность шага (секунды)
        :param L: колесная база (метры)
        :param weights: словарь весовых коэффициентов
        :param x0: начальное состояние (опционально)
        :param log_inner_newton: флаг логирования внутренних итераций Ньютона
        """
        self.ref_traj = ref_traj
        self.circles = circles
        self.dt = dt
        self.L = L
        self.weights = weights

        # Параметры задачи
        self.N = len(ref_traj) - 1  # количество шагов
        self.nx = 5  # размерность состояния
        self.nu = 2  # размерность управления

        # Размерность вектора переменных
        self.n_vars = self.N * (self.nx + self.nu) + self.nx

        # Размерность ограничений
        self.n_eq_constraints = self.N * self.nx  # динамика
        self.n_ineq_constraints = self.N + 1  # избегание столкновений
        self.n_constraints = self.n_eq_constraints + self.n_ineq_constraints

        # Начальное предположение
        if x0 is None:
            self.x0 = self._initialize_guess()
        else:
            self.x0 = x0

        # Логирование
        self.logger = IterationLogger(log_inner_newton=log_inner_newton)
        self.current_iteration = 0

    def _initialize_guess(self) -> np.ndarray:
        """
        Создает начальное предположение для оптимизации на основе опорной траектории.
        """
        X0 = np.zeros(self.n_vars)

        for i in range(self.N + 1):
            # Индекс текущего состояния
            idx = i * (self.nx + self.nu)

            # Заполняем состояние из опорной траектории
            X0[idx:idx + self.nx] = self.ref_traj[i]

            # Для управлений используем нули (можно улучшить)
            if i < self.N:
                X0[idx + self.nx:idx + self.nx + self.nu] = np.zeros(self.nu)

        return X0

    def objective(self, X: np.ndarray) -> float:
        """
        Целевая функция для оптимизации.
        """
        return cost_function(X, self.ref_traj, self.weights, self.N)

    def gradient(self, X: np.ndarray) -> np.ndarray:
        """
        Градиент целевой функции.
        """
        return cost_gradient(X, self.ref_traj, self.weights, self.N)

    def constraints(self, X: np.ndarray) -> np.ndarray:
        """
        Вектор ограничений.
        """
        # Ограничения динамики (равенства)
        dyn_constr = dynamics_constraints(
            X, self.dt, self.L, self.N, self.nx, self.nu)

        # Ограничения избегания столкновений (неравенства)
        coll_constr = collision_constraints(
            X, self.circles, self.N, self.nx, self.nu)

        return np.concatenate([dyn_constr, coll_constr])

    def jacobian(self, X: np.ndarray) -> np.ndarray:
        """
        Якобиан ограничений.
        """
        # Якобиан ограничений динамики
        dyn_jac = dynamics_constraints_jacobian(
            X, self.dt, self.L, self.N, self.nx, self.nu)

        # Якобиан ограничений избегания столкновений
        coll_jac = collision_constraints_jacobian(
            X, self.circles, self.N, self.nx, self.nu)

        # Объединяем якобианы
        return np.vstack([dyn_jac, coll_jac])

    def jacobianstructure(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Структура разреженности якобиана.
        """
        n_constr = self.n_constraints
        n_vars = self.n_vars
        rows, cols = np.meshgrid(
            np.arange(n_constr), np.arange(n_vars), indexing='ij')
        return rows.flatten(), cols.flatten()

    def intermediate(self, alg_mod, iter_count, obj_value, inf_pr, inf_du,
                     mu, d_norm, regularization_size, alpha_du, alpha_pr, ls_trials):
        """
        Callback-функция, вызываемая на каждой итерации IPOPT.
        """
        # Определяем тип итерации
        if alg_mod == 0:  # Внешняя итерация (μ-итерация)
            self.current_iteration = iter_count
            self.logger.start_outer_iteration(
                iter_count, mu, obj_value, inf_pr, inf_du,
                d_norm, alpha_du, alpha_pr, ls_trials
            )
            print(f"Outer iter {iter_count:3d}: μ={mu:.2e}, obj={obj_value:.6f}, "
                  f"inf_pr={inf_pr:.2e}, inf_du={inf_du:.2e}")

        elif alg_mod == 1 and self.logger.log_inner_newton:  # Внутренняя итерация Ньютона
            self.logger.add_inner_iteration(
                iter_count, obj_value, inf_pr, inf_du, d_norm,
                alpha_du, alpha_pr, ls_trials
            )
            print(f"  Inner iter {iter_count:3d}: obj={obj_value:.6f}, "
                  f"inf_pr={inf_pr:.2e}, inf_du={inf_du:.2e}")

        # Завершаем внешнюю итерацию при переходе к следующей
        if alg_mod == 0 and iter_count > 0:
            self.logger.end_outer_iteration()


def solve_ciao_problem_with_logging(problem: LoggingCIAOProblem,
                                    tol: float = 1e-6,
                                    max_iter: int = 1000,
                                    print_level: int = 5,
                                    log_filename: Optional[str] = None) -> Dict:
    """
    Решает задачу оптимизации траектории с использованием IPOPT и логированием.

    :param problem: экземпляр задачи LoggingCIAOProblem
    :param tol: допуск сходимости
    :param max_iter: максимальное количество итераций
    :param print_level: уровень вывода информации (0-12)
    :param log_filename: имя файла для сохранения лога (опционально)

    :return: словарь с результатами оптимизации
    """
    # Границы переменных
    n_vars = problem.n_vars
    lb = -np.inf * np.ones(n_vars)  # нижние границы
    ub = np.inf * np.ones(n_vars)   # верхние границы

    # Границы ограничений
    n_constr = problem.n_constraints
    cl = np.zeros(n_constr)         # нижние границы ограничений
    cu = np.zeros(n_constr)         # верхние границы ограничений

    # Для ограничений динамики (равенства): cl = cu = 0
    # Для ограничений столкновений (неравенства): cl = -inf, cu = 0
    cl[problem.n_eq_constraints:] = -np.inf
    cu[problem.n_eq_constraints:] = 0

    # Создаем решатель IPOPT
    nlp = ipopt.Problem(
        n=problem.n_vars,
        m=problem.n_constraints,
        problem_obj=problem,
        lb=lb,
        ub=ub,
        cl=cl,
        cu=cu
    )

    # Настраиваем параметры решателя
    nlp.add_option('tol', tol)
    nlp.add_option('max_iter', max_iter)
    nlp.add_option('print_level', print_level)
    nlp.add_option('mu_strategy', 'adaptive')
    nlp.add_option('hessian_approximation', 'limited-memory')

    # Включаем callback для промежуточных итераций
    nlp.add_option('output_file', 'ipopt_log.txt')

    print("Starting optimization with detailed logging...")
    start_time = time.time()

    # Решаем задачу
    X_opt, info = nlp.solve(problem.x0)

    end_time = time.time()

    # Завершаем последнюю итерацию
    problem.logger.end_outer_iteration(success=info['status'] == 0)

    # Сохраняем лог если указано имя файла
    if log_filename:
        problem.logger.save_to_file(log_filename)
        print(f"Optimization log saved to: {log_filename}")

    # Выводим статистику
    stats = problem.logger.get_summary_stats()
    print(f"\nOptimization completed in {end_time - start_time:.2f} seconds")
    print(f"Total outer iterations: {stats.get('total_outer_iterations', 0)}")
    print(f"Total inner iterations: {stats.get('total_inner_iterations', 0)}")
    print(f"Final μ: {stats.get('final_mu', 0):.2e}")
    print(f"Final objective: {stats.get('final_objective', 0):.6f}")

    return {
        'X_opt': X_opt,
        'info': info,
        'success': info['status'] == 0,
        'message': info['status_msg'],
        'logger': problem.logger,
        'stats': stats
    }


# Сохраняем старые функции для обратной совместимости
def extract_trajectory_from_solution(X_opt: np.ndarray,
                                     N: int,
                                     nx: int = 5,
                                     nu: int = 2) -> np.ndarray:
    """
    Извлекает траекторию состояний из вектора решения.
    """
    states = np.zeros((N + 1, nx))

    for i in range(N + 1):
        idx = i * (nx + nu)
        states[i] = X_opt[idx:idx + nx]

    return states


def extract_controls_from_solution(X_opt: np.ndarray,
                                   N: int,
                                   nx: int = 5,
                                   nu: int = 2) -> np.ndarray:
    """
    Извлекает последовательность управлений из вектора решения.
    """
    controls = np.zeros((N, nu))

    for i in range(N):
        idx = i * (nx + nu) + nx
        controls[i] = X_opt[idx:idx + nu]

    return controls
