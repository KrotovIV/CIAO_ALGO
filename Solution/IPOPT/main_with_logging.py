# main_with_logging.py

import numpy as np
from ciao_ipopt_solver_with_logging import solve_ciao_problem_with_logging, LoggingCIAOProblem
from ciao_ipopt_solver_with_logging import extract_trajectory_from_solution, extract_controls_from_solution
from reference_trajectory import generate_reference_trajectory
from occupancy_grid_utils import create_simple_occupancy_grid, occupancy_grid_to_circles
from visualization import plot_optimization_results
from obstacle_editor import create_obstacles_interactively
from optimization_visualizer import OptimizationVisualizer, load_optimization_log
from typing import Tuple
import matplotlib.pyplot as plt
import os
import datetime
from obstacle import CircularObstacle


def calculate_boundary_from_start_goal(start: Tuple[float, float],
                                       goal: Tuple[float, float],
                                       L: float = 2.5) -> Tuple[float, float, float, float]:
    """Вычисляет boundary square с запасом для маневров"""
    margin = L * 3

    min_x = min(start[0], goal[0]) - margin
    max_x = max(start[0], goal[0]) + margin
    min_y = min(start[1], goal[1]) - margin
    max_y = max(start[1], goal[1]) + margin

    # Гарантируем минимальный размер
    min_size = L * 6
    if max_x - min_x < min_size:
        center_x = (min_x + max_x) / 2
        min_x = center_x - min_size / 2
        max_x = center_x + min_size / 2

    if max_y - min_y < min_size:
        center_y = (min_y + max_y) / 2
        min_y = center_y - min_size / 2
        max_y = center_y + min_size / 2

    return (min_x, max_x, min_y, max_y)


def run_optimization_with_logging(obstacles, boundary_square, start, goal, dt, L, N, weights,
                                  log_inner_newton=True, log_filename=None):
    """Запускает процесс оптимизации с логированием."""
    print("\n=== Запуск оптимизации с логированием ===")
    print(f"Логирование внутренних итераций Ньютона: {log_inner_newton}")

    try:
        # Создаем Occupancy Grid
        occupancy_grid = create_simple_occupancy_grid(
            width=120, height=120, resolution=0.1,
            obstacles=obstacles
        )

        # Генерируем опорную траекторию
        ref_traj = generate_reference_trajectory(
            start, goal, occupancy_grid,
            grid_resolution=0.1,
            grid_origin=(0.0, 0.0),
            num_points=N+1,
            smooth=True
        )
        print("✓ Опорная траектория сгенерирована")

        # Преобразуем Occupancy Grid в ограничивающие круги
        circles, boundary_square = occupancy_grid_to_circles(
            occupancy_grid, ref_traj,
            grid_resolution=0.1,
            grid_origin=(0.0, 0.0),
            min_radius=0.8,
            dilation_radius=3,
            boundary_square=boundary_square
        )

        # Создаем задачу оптимизации с логированием
        problem = LoggingCIAOProblem(ref_traj, circles, dt, L, weights,
                                     log_inner_newton=log_inner_newton)

        # Генерируем имя файла для лога если не указано
        if log_filename is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = f"optimization_log_{timestamp}.json"

        # Решаем задачу с логированием
        result = solve_ciao_problem_with_logging(
            problem, tol=1e-6, max_iter=1000, print_level=5,
            log_filename=log_filename
        )

        if result['success']:
            print("✓ Оптимизация завершена успешно!")

            # Извлекаем результаты
            X_opt = result['X_opt']
            states_opt = extract_trajectory_from_solution(X_opt, N)
            controls_opt = extract_controls_from_solution(X_opt, N)

            # Визуализируем результаты оптимизации
            plot_optimization_results(
                boundary_square=boundary_square,
                states_opt=states_opt,
                controls_opt=controls_opt,
                ref_traj=ref_traj,
                circles=circles,
                occupancy_grid=occupancy_grid,
                grid_resolution=0.1,
                grid_origin=(0.0, 0.0),
                obstacles=obstacles,
                title="CIAO Trajectory Optimization with Logging",
                L=L
            )

            # Создаем визуализатор для процесса оптимизации
            log_data = {
                'log_timestamp': datetime.datetime.now().isoformat(),
                'log_inner_newton': log_inner_newton,
                'total_duration': result['stats']['total_duration'],
                'outer_iterations': result['logger'].outer_iterations
            }
            visualizer = OptimizationVisualizer(log_data)

            # Визуализируем процесс оптимизации
            visualizer.plot_convergence()
            plt.show()

            # Если есть внутренние итерации, показываем пример
            if log_inner_newton and len(visualizer.outer_iterations) > 0:
                # Показываем внутренние итерации для средней внешней итерации
                mid_iter = len(visualizer.outer_iterations) // 2
                fig = visualizer.plot_inner_iterations(mid_iter)
                if fig is not None:
                    plt.show()

            # Создаем анимацию сходимости
            print("Создание анимации сходимости...")
            visualizer.create_optimization_animation(
                f"convergence_animation_{timestamp}.gif")

            # Выводим сводку
            visualizer.print_summary()

            return True, log_filename

        else:
            print(f"✗ Ошибка оптимизации: {result['message']}")
            return False, None

    except Exception as e:
        print(f"✗ Ошибка при выполнении оптимизации: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def visualize_existing_log(log_filename: str):
    """
    Визуализирует существующий лог оптимизации.
    """
    if not os.path.exists(log_filename):
        print(f"Файл лога {log_filename} не найден")
        return

    try:
        visualizer = load_optimization_log(log_filename)

        # Строим графики сходимости
        fig = visualizer.plot_convergence()
        plt.show()

        # Если есть внутренние итерации, показываем пример
        if visualizer.log_data.get('log_inner_newton', False):
            if len(visualizer.outer_iterations) > 0:
                mid_iter = len(visualizer.outer_iterations) // 2
                fig = visualizer.plot_inner_iterations(mid_iter)
                if fig is not None:
                    plt.show()

        # Выводим сводку
        visualizer.print_summary()

    except Exception as e:
        print(f"Ошибка при загрузке лога: {e}")
        import traceback
        traceback.print_exc()


def run_complete_example_with_logging():
    """
    Запускает полный пример с логированием оптимизации.
    """
    print("=== CIAO Trajectory Optimization with Detailed Logging ===")
    print("Instructions:")
    print("- Edit obstacles in the editor window")
    print("- Choose logging options")
    print("- View optimization convergence plots and animations")

    # Параметры задачи
    dt = 0.1
    L = 2.5  # колесная база
    N = 30   # количество шагов

    # Весовые коэффициенты
    weights = {
        'wx': 10.0, 'wy': 10.0, 'wv': 1.0,
        'wtheta': 5.0, 'wdelta': 5.0,
        'wa': 0.1, 'womega': 0.1
    }

    # Определяем начальную и конечную точки
    start = (1.0, 1.0)
    goal = (10.0, 10.0)

    # Вычисляем boundary square
    boundary_square = calculate_boundary_from_start_goal(start, goal, L)

    # Инициализируем препятствия
    current_obstacles = []

    # Главный цикл программы
    while True:
        try:
            print(
                f"\nТекущее количество препятствий: {len(current_obstacles)}")
            print("Запуск редактора препятствий...")

            # Запускаем редактор
            result = create_obstacles_interactively(
                boundary_square=boundary_square,
                grid_resolution=0.1,
                initial_obstacles=current_obstacles,
                start=start,
                goal=goal,
                L=L/10
            )

            current_obstacles = result['obstacles']

            if result['action'] == 'optimize':
                # Запрашиваем настройки логирования
                print("\n=== Настройки логирования ===")
                log_inner_input = input(
                    "Логировать внутренние итерации Ньютона? (y/n, по умолчанию y): ").strip()
                log_inner = log_inner_input.lower() != 'n' if log_inner_input else True

                custom_log_name = input(
                    "Имя файла для лога (Enter для автоматического): ").strip()
                if not custom_log_name:
                    custom_log_name = None

                # Запускаем оптимизацию с логированием
                success, log_file = run_optimization_with_logging(
                    current_obstacles, boundary_square, start, goal, dt, L, N, weights,
                    log_inner_newton=log_inner, log_filename=custom_log_name
                )

                if success and log_file:
                    print(f"\n✓ Лог оптимизации сохранен в: {log_file}")

                    # Предлагаем визуализировать существующий лог
                    response = input(
                        "\nХотите визуализировать другой лог? (y/n): ").lower()
                    if response == 'y':
                        log_to_visualize = input(
                            "Введите имя файла лога: ").strip()
                        visualize_existing_log(log_to_visualize)

            elif result['action'] == 'exit':
                print("Выход из программы...")
                break

            elif result['action'] == 'close':
                # Окно редактора было закрыто - завершаем программу
                print("Редактор закрыт. Выход из программы...")
                break

        except KeyboardInterrupt:
            print("\nПрограмма прервана пользователем (Ctrl+C)")
            break
        except Exception as e:
            print(f"Неожиданная ошибка: {e}")
            import traceback
            traceback.print_exc()
            break


if __name__ == "__main__":
    run_complete_example_with_logging()
