"""
Барьерный метод для учёта ограничений.

F(x) = f(x) - μ * Σ log(-g_i(x))

Автор: Elizaveta Surda
Дата: 2026
"""

from collections.abc import Callable

import numpy as np

from src.bayesian_optimization import base


class BarrierMethod(base.ConstraintHandler):
    """
    Метод логарифмических барьерных функций.

    Атрибуты:
        constraint_functions: список функций g_i(x) <= 0
        barrier_coeff: коэффициент барьера μ
    """

    def __init__(
        self,
        constraint_functions: list[Callable[[np.ndarray], float]],
        barrier_coeff: float = 1.0,
    ) -> None:
        """
        Инициализация.

        Аргументы:
            constraint_functions: список функций ограничений g_i(x) <= 0
            barrier_coeff: коэффициент барьера μ
        """
        self.constraint_functions = constraint_functions
        self.barrier_coeff = barrier_coeff

    def evaluate_constraints(self, X: np.ndarray) -> np.ndarray:
        """
        Суммарное нарушение ограничений.

        Аргументы:
            X: точки, форма (n_points, n_dims)

        Возвращает:
            violations: форма (n_points,)
        """
        X = np.atleast_2d(X)
        violations = np.zeros(len(X))
        for constraint in self.constraint_functions:
            g_vals = np.array([constraint(x) for x in X])
            violations += np.maximum(0.0, g_vals)
        return violations

    def compute_penalized_objective(
        self, X: np.ndarray, f_values: np.ndarray
    ) -> np.ndarray:
        """
        F(x) = f(x) - μ * Σ log(-g_i(x))

        Аргументы:
            X: точки, форма (n_points, n_dims)
            f_values: значения целевой функции

        Возвращает:
            barrier_values: значения барьерной функции
        """
        X = np.atleast_2d(X)
        barrier = np.zeros(len(X))
        for constraint in self.constraint_functions:
            for j, x in enumerate(X):
                g = float(constraint(x))
                if g < -1e-10:
                    barrier[j] += -self.barrier_coeff * np.log(-g)
                elif g > 0:
                    barrier[j] = np.inf
        return f_values + barrier

    def get_acquisition_weights(self, X: np.ndarray) -> np.ndarray:
        """
        1 для допустимых точек, 0 для недопустимых.

        Аргументы:
            X: точки, форма (n_points, n_dims)

        Возвращает:
            weights: бинарные веса
        """
        return np.where(self.evaluate_constraints(X) <= 1e-6, 1.0, 0.0)

    def is_feasible(self, X: np.ndarray, tolerance: float = 1e-6) -> np.ndarray:
        """
        Проверка допустимости.

        Аргументы:
            X: точки, форма (n_points, n_dims)
            tolerance: допуск на нарушение

        Возвращает:
            feasible: булев массив
        """
        return self.evaluate_constraints(X) <= tolerance
