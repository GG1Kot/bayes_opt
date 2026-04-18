"""
Constrained Expected Improvement (CEI) acquisition функция.

CEI(x) = EI(x) * P(g(x) <= 0)

Автор: Elizaveta Surda
Дата: 2026
"""

from collections.abc import Callable

import numpy as np

from src.bayesian_optimization import base


class ConstrainedExpectedImprovement(base.ConstraintHandler):
    """
    CEI: произведение EI на вероятность допустимости.

    P(feasible) аппроксимируется сигмоидой:
        P = ∏ 1 / (1 + exp(β * g_i(x)))

    Атрибуты:
        constraint_functions: список функций g_i(x) <= 0
        xi: параметр exploration
    """

    def __init__(
        self,
        constraint_functions: list[Callable[[np.ndarray], float]],
        xi: float = 0.01,
    ) -> None:
        """
        Инициализация.

        Аргументы:
            constraint_functions: список функций ограничений g_i(x) <= 0
            xi: параметр баланса exploration/exploitation
        """
        self.constraint_functions = constraint_functions
        self.xi = xi

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
        CEI не штрафует целевую функцию — возвращает исходные значения.

        Аргументы:
            X: точки (не используются)
            f_values: значения целевой функции

        Возвращает:
            f_values: без изменений
        """
        return f_values

    def get_acquisition_weights(self, X: np.ndarray) -> np.ndarray:
        """
        Вероятность допустимости через сигмоиду.

        Аргументы:
            X: точки, форма (n_points, n_dims)

        Возвращает:
            prob: вероятности в [0, 1]
        """
        X = np.atleast_2d(X)
        prob = np.ones(len(X))
        beta = 5.0
        for constraint in self.constraint_functions:
            g_vals = np.array([constraint(x) for x in X])
            prob *= 1.0 / (1.0 + np.exp(beta * g_vals))
        return prob

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
