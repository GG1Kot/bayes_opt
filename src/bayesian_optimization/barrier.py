"""
Барьерный метод для учета ограничений.

Реализует метод логарифмических барьерных функций вида:
    F(x) = f(x) - μ * Σ log(-g_i(x))

для ограничений типа g_i(x) < 0. Требует строгой допустимости начальной точки.

Автор: Elizaveta Surda
Дата: 2026
"""

import numpy as np
from typing import List, Callable
from src.bayesian_optimization.base import ConstraintHandler


class BarrierMethod(ConstraintHandler):
    """
    Метод логарифмических барьерных функций.
    
    Создает барьер на границе допустимой области, не позволяя алгоритму
    выходить за пределы ограничений.
    
    Параметры:
        constraint_functions: список функций ограничений g_i(x) <= 0
        barrier_coeff: начальный коэффициент барьера μ (по умолчанию 1.0)
    
    Пример:
        >>> constraints = [lambda x: x[0] + x[1] - 1]
        >>> handler = BarrierMethod(constraints, barrier_coeff=1.0)
    """
    
    def __init__(self, constraint_functions: List[Callable], barrier_coeff: float = 1.0):
        """
        Инициализация барьерного метода.
        
        Параметры:
            constraint_functions: список функций ограничений
            barrier_coeff: коэффициент барьера
        """
        self.constraint_functions = constraint_functions
        self.barrier_coeff = barrier_coeff
    
    def evaluate_constraints(self, X: np.ndarray) -> np.ndarray:
        """
        Вычисление нарушения ограничений.
        
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
        Вычисление барьерной функции.
        
        F(x) = f(x) - μ * Σ log(-g_i(x))
        
        Параметры:
            X: точки, форма (n_points, n_dims)
            f_values: значения целевой функции
            
        Возвращает:
            barrier_values: значения барьерной функции
        """
        X = np.atleast_2d(X)
        barrier = np.zeros(len(X))
        
        for constraint in self.constraint_functions:
            for j, x in enumerate(X):
                g = constraint(x)
                if g < -1e-10:
                    barrier[j] += -self.barrier_coeff * np.log(-g)
                elif g > 0:
                    barrier[j] = np.inf
        
        return f_values + barrier
    
    def get_acquisition_weights(self, X: np.ndarray) -> np.ndarray:
        """
        Получение весов для acquisition функции.
        
        Для барьерного метода веса равны 1 только для допустимых точек.
        
        Параметры:
            X: точки для оценки
            
        Возвращает:
            weights: веса (1 для допустимых, 0 для недопустимых)
        """
        violations = self.evaluate_constraints(X)
        return np.where(violations <= 1e-6, 1.0, 0.0)
    
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