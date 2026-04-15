"""
Constrained Expected Improvement (CEI) acquisition функция.

Реализует acquisition функцию, учитывающую как улучшение целевой функции,
так и вероятность соблюдения ограничений:
    CEI(x) = EI(x) * P(g(x) <= 0)

Автор: Elizaveta Surda
Дата: 2026
"""

import numpy as np
from typing import List, Callable
from src.bayesian_optimization.base import ConstraintHandler


class ConstrainedExpectedImprovement(ConstraintHandler):
    """
    Constrained Expected Improvement acquisition функция.
    
    CEI(x) = E[max(f_best - f(x), 0)] * P(g(x) <= 0)
    
    Параметры:
        constraint_functions: список функций ограничений g_i(x) <= 0
        xi: параметр exploration (по умолчанию 0.01)
    
    Пример:
        >>> constraints = [lambda x: x[0] + x[1] - 1]
        >>> handler = ConstrainedExpectedImprovement(constraints, xi=0.01)
    """
    
    def __init__(self, constraint_functions: List[Callable], xi: float = 0.01):
        """
        Инициализация CEI acquisition функции.
        
        Параметры:
            constraint_functions: список функций ограничений
            xi: параметр exploration (trade-off)
        """
        self.constraint_functions = constraint_functions
        self.xi = xi
    
    def evaluate_constraints(self, X: np.ndarray) -> np.ndarray:
        """
        Вычисление суммарного нарушения ограничений.
        
        Параметры:
            X: точки для оценки, форма (n_points, n_dims)
            
        Возвращает:
            violations: массив нарушений
        """
        X = np.atleast_2d(X)
        violations = np.zeros(len(X))
        for constraint in self.constraint_functions:
            constraint_values = np.array([constraint(x) for x in X])
            violations += np.maximum(0, constraint_values)
        return violations
    
    def compute_penalized_objective(self, X: np.ndarray, f_values: np.ndarray) -> np.ndarray:
        """
        Для CEI штрафованная функция не используется.
        Возвращаем исходные значения.
        
        Параметры:
            X: точки
            f_values: значения целевой функции
            
        Возвращает:
            f_values: исходные значения
        """
        return f_values
    
    def get_acquisition_weights(self, X: np.ndarray) -> np.ndarray:
        """
        Вычисление вероятности допустимости точки.
        
        Использует сигмоидную функцию для аппроксимации:
            P(feasible) = ∏ 1/(1 + exp(β * g_i(x)))
        
        Параметры:
            X: точки для оценки
            
        Возвращает:
            prob: вероятности допустимости в диапазоне [0, 1]
        """
        X = np.atleast_2d(X)
        prob = np.ones(len(X))
        beta = 5.0
        
        for constraint in self.constraint_functions:
            g_vals = np.array([constraint(x) for x in X])
            prob *= 1 / (1 + np.exp(beta * g_vals))
        
        return prob
    
    def is_feasible(self, X: np.ndarray, tolerance: float = 1e-6) -> np.ndarray:
        """
        Проверка допустимости точек.
        
        Параметры:
            X: точки для проверки
            tolerance: допуск на нарушение
            
        Возвращает:
            feasible: булев массив
        """
        return self.evaluate_constraints(X) <= tolerance