"""
Метод штрафных функций для учета ограничений.

Реализует внешнюю штрафную функцию вида:
    F(x) = f(x) + ρ * Σ max(0, g_i(x))

где ρ - коэффициент штрафа. Чем больше ρ, тем точнее соблюдаются ограничения.

Автор: Elizaveta Surda
Дата: 2026
"""

import numpy as np
from typing import List, Callable
from src.bayesian_optimization.base import ConstraintHandler


class PenaltyMethod(ConstraintHandler):
    """
    Метод внешних штрафных функций.
    
    Преобразует задачу с ограничениями в задачу безусловной оптимизации
    путем добавления штрафа за нарушение ограничений.
    
    Параметры:
        constraint_functions: список функций ограничений g_i(x) <= 0
        penalty_coeff: коэффициент штрафа ρ (по умолчанию 100.0)
    
    Пример:
        >>> constraints = [lambda x: x[0] + x[1] - 1]
        >>> handler = PenaltyMethod(constraints, penalty_coeff=100.0)
    """
    
    def __init__(self, constraint_functions: List[Callable], penalty_coeff: float = 100.0):
        """
        Инициализация метода штрафных функций.
        
        Параметры:
            constraint_functions: список функций ограничений
            penalty_coeff: коэффициент штрафа
        """
        self.constraint_functions = constraint_functions
        self.penalty_coeff = penalty_coeff
    
    def evaluate_constraints(self, X: np.ndarray) -> np.ndarray:
        """
        Вычисление суммарного нарушения ограничений.
        
        Параметры:
            X: точки для оценки, форма (n_points, n_dims)
            
        Возвращает:
            violations: массив нарушений, форма (n_points,)
        """
        X = np.atleast_2d(X)
        violations = np.zeros(len(X))
        for constraint in self.constraint_functions:
            constraint_values = np.array([constraint(x) for x in X])
            violations += np.maximum(0, constraint_values)
        return violations
    
    def compute_penalized_objective(self, X: np.ndarray, f_values: np.ndarray) -> np.ndarray:
        """
        Вычисление штрафованной целевой функции.
        
        F(x) = f(x) + ρ * Σ max(0, g_i(x))
        
        Параметры:
            X: точки, форма (n_points, n_dims)
            f_values: значения целевой функции, форма (n_points,)
            
        Возвращает:
            penalized: значения штрафованной функции
        """
        violations = self.evaluate_constraints(X)
        return f_values + self.penalty_coeff * violations
    
    def get_acquisition_weights(self, X: np.ndarray) -> np.ndarray:
        """
        Получение весов для acquisition функции.
        
        Веса обратно пропорциональны нарушению ограничений.
        
        Параметры:
            X: точки для оценки, форма (n_points, n_dims)
            
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