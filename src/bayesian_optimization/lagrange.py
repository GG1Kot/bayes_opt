"""
Метод множителей Лагранжа для задач с ограничениями.

Реализует Augmented Lagrangian метод (ALM), который комбинирует
штрафной метод и метод множителей Лагранжа:
    L(x, λ, ρ) = f(x) + Σ λ_i * g_i(x) + (ρ/2) * Σ max(0, g_i(x))^2

Автор: Elizaveta Surda
Дата: 2026
"""

import numpy as np
from typing import List, Callable
from src.bayesian_optimization.base import ConstraintHandler


class LagrangeMethod(ConstraintHandler):
    """
    Augmented Lagrangian метод (ALM).
    
    Использует оценки множителей Лагранжа для более точного учета
    ограничений по сравнению с простым штрафным методом.
    
    Параметры:
        constraint_functions: список функций ограничений g_i(x) <= 0
        penalty_coeff: начальный коэффициент штрафа ρ (по умолчанию 10.0)
    
    Пример:
        >>> constraints = [lambda x: x[0] + x[1] - 1]
        >>> handler = LagrangeMethod(constraints, penalty_coeff=10.0)
    """
    
    def __init__(self, constraint_functions: List[Callable], penalty_coeff: float = 10.0):
        """
        Инициализация Augmented Lagrangian метода.
        
        Параметры:
            constraint_functions: список функций ограничений
            penalty_coeff: начальный коэффициент штрафа
        """
        self.constraint_functions = constraint_functions
        self.penalty_coeff = penalty_coeff
        self.multipliers = np.zeros(len(constraint_functions))
    
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
        Вычисление Augmented Lagrangian функции.
        
        L(x, λ, ρ) = f(x) + Σ λ_i * g_i(x) + (ρ/2) * Σ max(0, g_i(x))^2
        
        Параметры:
            X: точки, форма (n_points, n_dims)
            f_values: значения целевой функции
            
        Возвращает:
            lagrangian: значения Augmented Lagrangian
        """
        X = np.atleast_2d(X)
        result = f_values.copy()
        
        for i, constraint in enumerate(self.constraint_functions):
            g_vals = np.array([constraint(x) for x in X])
            violations = np.maximum(0, g_vals)
            result += self.multipliers[i] * violations + 0.5 * self.penalty_coeff * violations**2
        
        return result
    
    def get_acquisition_weights(self, X: np.ndarray) -> np.ndarray:
        """
        Получение весов для acquisition функции.
        
        Параметры:
            X: точки для оценки
            
        Возвращает:
            weights: веса в диапазоне [0, 1]
        """
        violations = self.evaluate_constraints(X)
        return np.exp(-self.penalty_coeff * violations)
    
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