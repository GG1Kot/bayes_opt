"""
Augmented Lagrangian метод для учёта ограничений.

L(x, λ, ρ) = f(x) + Σ λ_i * g_i(x) + (ρ/2) * Σ max(0, g_i(x))^2

Автор: Elizaveta Surda
Дата: 2026
"""

from collections.abc import Callable

import numpy as np

from src.bayesian_optimization import base


class LagrangeMethod(base.ConstraintHandler):
    """
    Augmented Lagrangian метод (ALM).

    Атрибуты:
        constraint_functions: список функций g_i(x) <= 0
        penalty_coeff: коэффициент штрафа ρ
        multipliers: текущие оценки множителей Лагранжа λ_i
    """

    def __init__(
        self,
        constraint_functions: list[Callable[[np.ndarray], float]],
        penalty_coeff: float = 10.0,
    ) -> None:
        """
        Инициализация.

        Аргументы:
            constraint_functions: список функций ограничений g_i(x) <= 0
            penalty_coeff: начальный коэффициент штрафа ρ
        """
        self.constraint_functions = constraint_functions
        self.penalty_coeff = penalty_coeff
        self.multipliers = np.zeros(len(constraint_functions))

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
        L(x, λ, ρ) = f(x) + Σ λ_i*g_i(x) + (ρ/2)*Σ max(0,g_i(x))^2

        Аргументы:
            X: точки, форма (n_points, n_dims)
            f_values: значения целевой функции

        Возвращает:
            lagrangian: значения Augmented Lagrangian
        """
        X = np.atleast_2d(X)
        result = f_values.copy()
        for i, constraint in enumerate(self.constraint_functions):
            g_vals = np.array([constraint(x) for x in X])
            violations = np.maximum(0.0, g_vals)
            result += (
                self.multipliers[i] * violations
                + 0.5 * self.penalty_coeff * violations**2
            )
        return result

    def get_acquisition_weights(self, X: np.ndarray) -> np.ndarray:
        """
        Экспоненциальное затухание при нарушении ограничений.

        Аргументы:
            X: точки, форма (n_points, n_dims)

        Возвращает:
            weights: веса в [0, 1]
        """
        return np.exp(-self.penalty_coeff * self.evaluate_constraints(X))

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
