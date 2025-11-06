import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
import json
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec


class OptimizationVisualizer:
    """
    Класс для визуализации процесса оптимизации.
    """

    def __init__(self, log_data: Dict):
        self.log_data = log_data
        self.outer_iterations = log_data.get('outer_iterations', [])

    def plot_convergence(self, figsize: Tuple[float, float] = (12, 8)):
        """
        Строит графики сходимости оптимизации.
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('IPOPT Optimization Convergence', fontsize=16)

        # Подготовка данных
        iterations = [iter_data['iter'] for iter_data in self.outer_iterations]
        mu_values = [iter_data['mu'] for iter_data in self.outer_iterations]
        obj_values = [iter_data['obj_value']
                      for iter_data in self.outer_iterations]
        inf_pr_values = [iter_data['inf_pr']
                         for iter_data in self.outer_iterations]
        inf_du_values = [iter_data['inf_du']
                         for iter_data in self.outer_iterations]

        # 1. Параметр барьера μ
        ax1 = axes[0, 0]
        ax1.semilogy(iterations, mu_values, 'bo-', linewidth=2, markersize=4)
        ax1.set_xlabel('Outer Iteration')
        ax1.set_ylabel('Barrier Parameter μ')
        ax1.set_title('Barrier Parameter Convergence')
        ax1.grid(True, alpha=0.3)

        # 2. Целевая функция
        ax2 = axes[0, 1]
        ax2.plot(iterations, obj_values, 'ro-', linewidth=2, markersize=4)
        ax2.set_xlabel('Outer Iteration')
        ax2.set_ylabel('Objective Value')
        ax2.set_title('Objective Function Convergence')
        ax2.grid(True, alpha=0.3)

        # 3. Нарушения ограничений
        ax3 = axes[1, 0]
        ax3.semilogy(iterations, inf_pr_values, 'go-',
                     linewidth=2, markersize=4, label='Primal')
        ax3.semilogy(iterations, inf_du_values, 'mo-',
                     linewidth=2, markersize=4, label='Dual')
        ax3.set_xlabel('Outer Iteration')
        ax3.set_ylabel('Constraint Violation')
        ax3.set_title('Constraint Violation Convergence')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Внутренние итерации (если есть)
        ax4 = axes[1, 1]
        inner_counts = []
        # ax4.bar(iterations, inner_counts, alpha=0.7, color='orange')
        ax4.set_xlabel('Outer Iteration')
        ax4.set_ylabel('Number of Inner Iterations')
        ax4.set_title('Inner Newton Iterations per Outer Iteration')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_inner_iterations(self, outer_iter_index: int, figsize: Tuple[float, float] = (10, 6)):
        """
        Строит графики внутренних итераций Ньютона для конкретной внешней итерации.
        """
        if outer_iter_index >= len(self.outer_iterations):
            print(f"Warning: Outer iteration {outer_iter_index} not found")
            return None

        outer_iter = self.outer_iterations[outer_iter_index]
        inner_iters = outer_iter.get('inner_iterations', [])

        if not inner_iters:
            print(
                f"No inner iterations logged for outer iteration {outer_iter_index}")
            return None

        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(f'Inner Newton Iterations (Outer Iteration {outer_iter_index}, μ={outer_iter["mu"]:.2e})',
                     fontsize=14)

        inner_iter_nums = [iter_data['inner_iter']
                           for iter_data in inner_iters]
        obj_values = [iter_data['obj_value'] for iter_data in inner_iters]
        inf_pr_values = [iter_data['inf_pr'] for iter_data in inner_iters]
        inf_du_values = [iter_data['inf_du'] for iter_data in inner_iters]
        lg_norm_values = [iter_data['lg_norm'] for iter_data in inner_iters]

        # 1. Целевая функция
        ax1 = axes[0, 0]
        ax1.plot(inner_iter_nums, obj_values, 'bo-', linewidth=2, markersize=4)
        ax1.set_xlabel('Inner Iteration')
        ax1.set_ylabel('Objective Value')
        ax1.set_title('Objective Function')
        ax1.grid(True, alpha=0.3)

        # 2. Нарушения ограничений
        ax2 = axes[0, 1]
        ax2.semilogy(inner_iter_nums, inf_pr_values, 'go-',
                     linewidth=2, markersize=4, label='Primal')
        ax2.semilogy(inner_iter_nums, inf_du_values, 'mo-',
                     linewidth=2, markersize=4, label='Dual')
        ax2.set_xlabel('Inner Iteration')
        ax2.set_ylabel('Constraint Violation')
        ax2.set_title('Constraint Violation')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Норма Лагранжиана
        ax3 = axes[1, 0]
        ax3.semilogy(inner_iter_nums, lg_norm_values,
                     'co-', linewidth=2, markersize=4)
        ax3.set_xlabel('Inner Iteration')
        ax3.set_ylabel('Lagrangian Norm')
        ax3.set_title('Lagrangian Norm')
        ax3.grid(True, alpha=0.3)

        # 4. Шаги
        ax4 = axes[1, 1]
        alpha_pr_values = [iter_data['alpha_pr'] for iter_data in inner_iters]
        alpha_du_values = [iter_data['alpha_du'] for iter_data in inner_iters]
        ax4.plot(inner_iter_nums, alpha_pr_values, 'yo-',
                 linewidth=2, markersize=4, label='Primal Step')
        ax4.plot(inner_iter_nums, alpha_du_values, 'ko-',
                 linewidth=2, markersize=4, label='Dual Step')
        ax4.set_xlabel('Inner Iteration')
        ax4.set_ylabel('Step Size')
        ax4.set_title('Step Sizes')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def create_optimization_animation(self, filename: str = 'optimization_convergence.gif',
                                      fps: int = 2):
        """
        Создает анимацию процесса сходимости.
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        def update(frame):
            for ax in axes.flat:
                ax.clear()

            # Текущие данные до frame-й итерации
            current_data = self.outer_iterations[:frame+1]
            if not current_data:
                return axes.flat

            iterations = [iter_data['iter'] for iter_data in current_data]
            mu_values = [iter_data['mu'] for iter_data in current_data]
            obj_values = [iter_data['obj_value'] for iter_data in current_data]
            inf_pr_values = [iter_data['inf_pr'] for iter_data in current_data]
            inf_du_values = [iter_data['inf_du'] for iter_data in current_data]

            # 1. Параметр барьера μ
            axes[0, 0].semilogy(iterations, mu_values,
                                'bo-', linewidth=2, markersize=4)
            axes[0, 0].set_xlabel('Outer Iteration')
            axes[0, 0].set_ylabel('Barrier Parameter μ')
            axes[0, 0].set_title('Barrier Parameter Convergence')
            axes[0, 0].grid(True, alpha=0.3)

            # 2. Целевая функция
            axes[0, 1].plot(iterations, obj_values, 'ro-',
                            linewidth=2, markersize=4)
            axes[0, 1].set_xlabel('Outer Iteration')
            axes[0, 1].set_ylabel('Objective Value')
            axes[0, 1].set_title('Objective Function Convergence')
            axes[0, 1].grid(True, alpha=0.3)

            # 3. Нарушения ограничений
            axes[1, 0].semilogy(iterations, inf_pr_values,
                                'go-', linewidth=2, markersize=4, label='Primal')
            axes[1, 0].semilogy(iterations, inf_du_values,
                                'mo-', linewidth=2, markersize=4, label='Dual')
            axes[1, 0].set_xlabel('Outer Iteration')
            axes[1, 0].set_ylabel('Constraint Violation')
            axes[1, 0].set_title('Constraint Violation Convergence')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

            # 4. Информация
            axes[1, 1].text(0.1, 0.8, f'Iteration: {frame}', fontsize=12)
            axes[1, 1].text(0.1, 0.6, f'μ: {mu_values[-1]:.2e}', fontsize=12)
            axes[1, 1].text(
                0.1, 0.4, f'Objective: {obj_values[-1]:.6f}', fontsize=12)
            axes[1, 1].text(
                0.1, 0.2, f'Primal Inf: {inf_pr_values[-1]:.2e}', fontsize=12)
            axes[1, 1].set_xlim(0, 1)
            axes[1, 1].set_ylim(0, 1)
            axes[1, 1].set_title('Current Status')
            axes[1, 1].axis('off')

            fig.suptitle(
                f'IPOPT Optimization Progress (Iteration {frame})', fontsize=16)
            return axes.flat

        anim = FuncAnimation(fig, update, frames=len(self.outer_iterations),
                             interval=1000//fps, blit=False)

        anim.save(filename, writer='pillow', fps=fps)
        plt.close()
        print(f"Optimization animation saved as {filename}")

    def print_summary(self):
        """
        Выводит сводную информацию по оптимизации.
        """
        stats = {
            'total_outer_iterations': len(self.outer_iterations),
            'total_inner_iterations': sum(len(iter_data.get('inner_iterations', []))
                                          for iter_data in self.outer_iterations),
            'final_mu': self.outer_iterations[-1]['mu'] if self.outer_iterations else 0,
            'final_objective': self.outer_iterations[-1]['obj_value'] if self.outer_iterations else 0,
            'total_duration': self.log_data.get('total_duration', 0),
            'log_inner_newton': self.log_data.get('log_inner_newton', False)
        }

        print("\n" + "="*50)
        print("OPTIMIZATION SUMMARY")
        print("="*50)
        print(f"Total outer iterations: {stats['total_outer_iterations']}")
        print(f"Total inner iterations: {stats['total_inner_iterations']}")
        print(f"Final barrier parameter μ: {stats['final_mu']:.2e}")
        print(f"Final objective value: {stats['final_objective']:.6f}")
        print(f"Total duration: {stats['total_duration']:.2f} seconds")
        print(f"Inner Newton iterations logged: {stats['log_inner_newton']}")
        print("="*50)


def load_optimization_log(filename: str) -> OptimizationVisualizer:
    """
    Загружает лог оптимизации из файла и создает визуализатор.
    """
    with open(filename, 'r') as f:
        log_data = json.load(f)

    return OptimizationVisualizer(log_data)
